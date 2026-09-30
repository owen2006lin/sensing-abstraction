from src.load import *
from src.angular_error_2 import *
from src.range_2 import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


GROUP_COLS = ["scenario_id", "drop_id"]
tags, labels, features = load_pos()


groups = tags.select(pl.struct(GROUP_COLS).hash()).to_series().to_numpy()
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(gss.split(features, labels, groups=groups)) 
X,     y,     tags_tr  = features[train_idx], labels[train_idx], tags[train_idx]
X_val, y_val, tags_val = features[val_idx],   labels[val_idx],   tags[val_idx]
 


#==========================================Angular Error==========================#
'''
edge=-1.195
close=2.490
rss=-0.082
intercept=-16.306
'''
slip_labels, slip_tags, slip_features = build_slip_features(X, y)
slip_features = build_edge_close(slip_features)


slip_model = train_slip_model(slip_features, slip_labels, print_params = True)


angular_errors = angular_error(slip_model, X_val)

#==========================================Range Error===========================#

regime_labels = label_regime(y)["region"]
#oof_logits, oof_cal, b_final, models = predict_out_of_fold(X, regime_labels, tags_tr, GBM_PARAMS)
#save_class_outputs(oof_logits, oof_cal, b_final, models, "test_range_1")

# bulk fitting
range_errors = y["range_error_m"]
bulk = X.with_columns(regime_labels, range_errors).filter(pl.col("region") == 0)
bulk_model = fit_bulk(bulk, print_stats = True)
bulk_distribution = build_residual_distribution(bulk_model, bulk.select("bistatic_range_m"), bulk["range_error_m"])

# non bulk distribution
[sb_plus, sb_minus, far_plus, far_minus, near] = build_distributions(X,label_regime(y),regions = ["sideband+", "sideband-", "far+", "far-", "near"])
non_bulk_distributions = [sb_plus, sb_minus, far_plus, far_minus, near]

#sampling
oof_logits, oof_cal, b_final, models = load_class_model("test_range_1")
regime_probs = predict_regime(X_val, b_final, models)
classes = sample_class(regime_probs)

sample_features = X_val.select(
    pl.col("nn_norm_range_sep"),
    pl.col("bistatic_range_m")
).with_columns(classes)

#bulk
bulk_sample_features = sample_features.filter(pl.col("regime") == 0)
bulk_samples = sample_bulk(bulk_sample_features, bulk_model, bulk_distribution)

non_bulk_features = (sample_features.filter(pl.col("regime")!=0)
                     .select(["nn_norm_range_sep" ,"regime"])
                     .fill_null(float("inf")))
nn_dist = non_bulk_features.select("nn_norm_range_sep").to_numpy().ravel()
regime = non_bulk_features.select("regime").to_numpy().ravel().astype(int)
non_bulk_features = list(zip(regime, nn_dist))

non_bulk_samples = sample_non_bulk(non_bulk_features, non_bulk_distributions)

#put it back together in order
mask = (sample_features["regime"] == 0).to_numpy()
non_bulk_df = pl.DataFrame({"sampled_error" : non_bulk_samples})
combined = pl.concat([bulk_samples, non_bulk_df], how="vertical")
order = np.empty(len(mask), dtype=int)
order[mask] = np.arange(mask.sum())  
order[~mask] = mask.sum() + np.arange((~mask).sum())
merged = combined[order]
#==========================================Full Pos Error==========================#
errors = pl.concat([merged, angular_errors.select(["az_err", "el_err"])], how = "horizontal")
print(errors)
extra = (X_val.with_columns((pl.col("bistatic_range_m") / 2).alias("range"))
              .select(["range", "rx_elevation_deg"])
         )
conversion = pl.concat([errors, extra], how = "horizontal")


def add_3d_position_error(
    df: pl.DataFrame,
    range_err_col: str = "sampled_error",
    az_err_col: str = "az_err",
    el_err_col: str = "el_err",
    range_col: str = "range",
    el_col: str = "rx_elevation_deg",
    angle_unit: str = "deg",  # "deg", "rad" or "mrad" for az_err / el_err
) -> pl.DataFrame:
    """Add 3D position error columns computed from range, az and el errors.
 
    Adds these columns, in the same units as `range`:
      err_along   - error along the line of sight (the range error)
      err_cross_h - horizontal cross-range error, R*cos(el)*d_az
      err_cross_v - vertical cross-range error, R*d_el
      err_3d      - exact 3D error: distance between the true point and the
                    perturbed point (accurate for any size of angle error)
      err_3d_approx - small-angle RSS of the three components above
    """
    scale = {"deg": 3.141592653589793 / 180, "rad": 1.0, "mrad": 1e-3}[angle_unit]
 
    R = pl.col(range_col)
    el = pl.col(el_col).radians()          # elevation column is in degrees
    d_r = pl.col(range_err_col)
    d_az = pl.col(az_err_col) * scale
    d_el = pl.col(el_err_col) * scale
 
    # Exact: put the true point at az = 0 (the 3D error doesn't depend on the
    # absolute azimuth), convert both points to Cartesian and take the distance.
    R2, el2 = R + d_r, el + d_el
    dx = R2 * el2.cos() * d_az.cos() - R * el.cos()
    dy = R2 * el2.cos() * d_az.sin()
    dz = R2 * el2.sin() - R * el.sin()
 
    return df.with_columns(
        err_along=d_r,
        err_cross_h=R * el.cos() * d_az,
        err_cross_v=R * d_el,
        err_3d=(dx**2 + dy**2 + dz**2).sqrt(),
    ).with_columns(
        err_3d_approx=(
            pl.col("err_along") ** 2
            + pl.col("err_cross_h") ** 2
            + pl.col("err_cross_v") ** 2
        ).sqrt()
    )

pos_error = add_3d_position_error(conversion)
print(pos_error)

fig = go.Figure()

fig.add_trace(go.Histogram(
    x=pos_error["err_3d"].to_numpy().ravel(),
    nbinsx=100,
    name="sampled 3d error - exact",
    opacity=0.6
))

fig.add_trace(go.Histogram(
    x=pos_error["err_3d_approx"].to_numpy().ravel(),
    nbinsx=100,
    name="sampled 3d error - small angle approximation",
    opacity=0.6
))

fig.add_trace(go.Histogram(
    x=y["position_error_3d_m"].to_numpy().ravel(),
    nbinsx=100,
    name="labeled error",
    opacity=0.6
))
fig.add_trace(go.Histogram(
    x=y_val["position_error_3d_m"].to_numpy().ravel(),
    nbinsx=100,
    name="labeled error",
    opacity=0.6
))

fig.update_layout(
    barmode="overlay",
    xaxis_title="value",
    yaxis_title="count",
    legend_title="series"
)

fig.show()