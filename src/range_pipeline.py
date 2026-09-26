from src.range import *
from sklearn.model_selection import GroupShuffleSplit
from scipy.special import softmax
from scipy.stats import poisson
 
#================================ Load + label ================================#
tags, errors, features = load_pos()
 
REGION_NAMES = ["bulk", "sideband+", "sideband-", "far+", "far-", "near"]   # index = label 0..5
e = pl.col("range_error_m")
labels = (
    errors.select("range_error_m")
    .with_columns(
        pl.when(e.is_between(-0.52, 0.82)).then(0)
        .when(e.is_between(1.40, 1.85)).then(1)
        .when(e.is_between(-2.25, -1.55)).then(2)
        .when(e > 1.85).then(3)
        .when(e < -2.25).then(4)
        .otherwise(5)
        .cast(pl.Int32)
        .alias("region")
    )
    .with_columns(pl.col("region").replace_strict(list(range(6)), REGION_NAMES).alias("region_txt"))
)
 
#======================= Grouped train / validation split =====================#
groups = tags.select(pl.struct(GROUP_COLS).hash()).to_series().to_numpy()
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(gss.split(features, labels, groups=groups))
 
X,     y,     tags_tr  = features[train_idx], labels[train_idx], tags[train_idx]
X_val, y_val, tags_val = features[val_idx],   labels[val_idx],   tags[val_idx]
 
#============================ Train + cache to disk ===========================#
#out_of_fold, oof_cal, b_final, models = predict_out_of_fold(X, y["region"], tags_tr, GBM_PARAMS)
#np.savetxt("data/feature_cache/oof_class_preds.csv", oof_cal)
#np.savetxt("data/feature_cache/b_final.csv", b_final)
#np.savetxt("data/feature_cache/oof.csv", out_of_fold)
#for i, model in enumerate(models):
#    model.save_model(f"models/regime_class_{i}.txt")

# Probability outputs
oof_cal = np.genfromtxt("data/feature_cache/oof_class_preds.csv")

# Bias, raw logits
b_final = np.genfromtxt("data/feature_cache/b_final.csv")

# Scores with raw logits
out_of_fold = np.genfromtxt("data/feature_cache/oof.csv")
classes = sample_class(oof_cal)


pred = X.select(
    pl.col("nn_norm_range_sep"),
    pl.col("bistatic_range_m")
).with_columns(classes)
#===================================Distribution building====================#
bulk = pl.concat([features, labels], how = "horizontal").filter(pl.col("region_txt") == "bulk")
bulk_model = fit_bulk(bulk)
bulk_distribution = build_residual_distribution(bulk_model, bulk.select("bistatic_range_m"), bulk["range_error_m"])

bulk_inputs = pred.filter(
    (pl.col("regime") == 0)
)
bulk_outputs = bulk_inputs.with_columns(
    pl.col("bistatic_range_m")
      .map_elements(lambda r: float(np.ravel(sample_bulk(bulk_model, r, bulk_distribution))[0]),
                    return_dtype=pl.Float64)
      .alias("sampled_error")
).select("sampled_error")



distributions = build_distributions(features,labels,regions = ["sideband+", "sideband-", "far+", "far-", "near"])

errors = []
non_bulk = pred.filter(
    pl.col("regime") != 0
)
data = []
for val, _, regime in non_bulk.iter_rows():
    if val is None:
        data.append((regime, float('inf')))
    else:
        data.append((regime, val))

non_bulk_preds = sample_non_bulk(data, distributions)
print(len(non_bulk_preds))
print(len(bulk_outputs))



x = np.concatenate([non_bulk_preds, bulk_outputs.to_numpy().ravel()])
import numpy as np
import plotly.express as px

fig = px.histogram(x=x, nbins=50)
fig.show()

fig = px.histogram(x=y["range_error_m"].to_numpy().ravel(), nbins=50)
fig.show()