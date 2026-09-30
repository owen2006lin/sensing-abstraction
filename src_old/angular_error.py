from src.load import *
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
'''
First: computes angular error through:
range
u/v for azimuth + elevation

First get az el -> u v
Also get frac_u, frac_v

Then subtract
u - frac_u
v - frac_v

sample slip_u, slip_v binned by frac (high dependence on these guys

convert u_est, v_est -> az_est, el_est
subtract

'''
''''''
tags, labels, features = load_pos()
features.write_csv("slip_test.csv")

#------------------------------------------Angular Error--------------------------------------#
# (No sampling for now)
from src.feature_builder import *
NGRID = 32.0
AZ0 = 30.0

sim = features.select(
    pl.col("u_true"),
    pl.col("v_true"),
    ((NGRID * pl.col("u_true") - pl.col("frac_u"))/ NGRID).alias("u_est"),
    ((NGRID * pl.col("v_true") - pl.col("frac_v"))/ NGRID).alias("v_est"),
    pl.col("rx_azimuth_deg"),
    pl.col("rx_elevation_deg"),
    pl.col("slip_v"),
    pl.col("slip_u"),
    pl.col("frac_u"),
    pl.col("frac_v")
)


az_est, el_est = inv_uv("u_est", "v_est")
sim = sim.with_columns(
    az_est.alias("az_est"),
    el_est.alias("el_est"),
    (pl.col("u_true") - pl.col("u_est")).alias("u_error"),
    (pl.col("v_true") - pl.col("v_est")).alias("v_error"),
).with_columns(
    (pl.col("az_est") - pl.col("rx_azimuth_deg")).alias("az_err"),
    (pl.col("el_est") - pl.col("rx_elevation_deg")).alias("el_err"),
)
#=================================Fitting Slippage Sampler=======================#
def slip_stats(df: pl.DataFrame, max_mag: int = 4) -> dict:
    """p_toward and mag_dist for u, v, and pooled (u+v stacked, as in the fit script)."""
    long = pl.concat([
        df.select(axis=pl.lit(ax), slip=pl.col(f"slip_{ax}"), frac=pl.col(f"frac_{ax}"))
        for ax in ("u", "v")
    ])
    slips = long.filter(pl.col("slip") != 0)

    def stats(s: pl.DataFrame) -> dict:
        ones = s.filter(pl.col("slip").abs() == 1)
        p_toward = ones.select((pl.col("slip").sign() == pl.col("frac").sign()).mean()).item()
        mag = (s.group_by(mag=pl.col("slip").abs().clip(upper_bound=max_mag))
                 .len().sort("mag")
                 .with_columns(p=pl.col("len") / pl.col("len").sum()))
        return {"n_slips": s.height, "p_toward": p_toward,
                "mag_dist": dict(zip(mag["mag"].to_list(), mag["p"].to_list()))}

    out = {ax: stats(slips.filter(pl.col("axis") == ax)) for ax in ("u", "v")}
    out["pooled"] = stats(slips)
    return out

def print_slip_stats(stats: dict, max_mag: int = 4) -> None:
    mags = sorted({int(m) for s in stats.values() for m in s["mag_dist"]})
    labels = [f"{m}+" if m == max_mag else str(m) for m in mags]
    print(f"{'axis':<8}{'n_slips':>8}{'p_toward':>10}  " + "".join(f"{'|' + l + '|':>8}" for l in labels))
    for ax, s in stats.items():
        dist = {int(k): v for k, v in s["mag_dist"].items()}
        p = s["p_toward"]
        p_str = f"{p:.3f}" if p is not None else "-"
        print(f"{ax:<8}{s['n_slips']:>8}{p_str:>10}  " + "".join(f"{dist.get(m, 0):>8.3f}" for m in mags))


'''
axis     n_slips  p_toward       |1|     |2|     |3|    |4+|
u            679     0.921     0.934   0.043   0.015   0.009
v            584     0.951     0.935   0.053   0.007   0.005
pooled      1263     0.935     0.934   0.048   0.011   0.007

Reasonable to assume u,v slippage are largely identical -> shared model to predict slipping
'''

EDGE_MIN = 1e-3  

slip_label_u = sim.select(pl.col("slip_u").alias("slip"))
slip_features_u = (features.select(["rss_max", "nn_norm_range_sep", "abs_frac_u"]) 
                            .rename({"abs_frac_u" : "abs_frac"})
                            .with_columns(
                                pl.lit("u").alias("type")
                            )
                    )


slip_label_v = sim.select(pl.col("slip_v").alias("slip"))
slip_features_v = (features.
                            select(["rss_max", "nn_norm_range_sep","abs_frac_v"])
                            .rename({"abs_frac_v" : "abs_frac"})
                            .with_columns(
                                pl.lit("v").alias("type")
                            )
                    )

slip_labels = slip_label_u.vstack(slip_label_v)
slip_features = slip_features_u.vstack(slip_features_v)
slip_tags = slip_features.select("type")
slip_features = slip_features.drop("type")

# Build edge, rss, close features
# Pad out nulls to max value since log reg can't take nulls
LARGE_VALUE = slip_features["nn_norm_range_sep"].max()
print(LARGE_VALUE)
slip_features = slip_features.with_columns(
    pl.col("nn_norm_range_sep").fill_null(LARGE_VALUE)
)
slip_features = slip_features.with_columns(
    (0.5 - pl.col("abs_frac") + EDGE_MIN).log().alias("edge"),
    (pl.col("nn_norm_range_sep") / (-5)).exp().alias("close"),
    (pl.col("rss_max")).alias("rss")
).select(
    ["edge", "close", "rss"]
)


# Labels: slip
slip_labels = slip_labels.select(
    (pl.col("slip") != 0).alias("slipped")
).select("slipped")

print(slip_labels)
print(slip_features)    


model = LogisticRegression(C = 10, max_iter=1000).fit(slip_features, slip_labels["slipped"])
b0 = model.intercept_[0]
b1, b2, b3 = model.coef_[0]

for name, c in zip(slip_features.columns, model.coef_[0]):
    print(f"{name}={c:.3f}")
print(f"intercept={model.intercept_[0]:.3f}")




SLIP_MAG_ODDS = [0.933, 0.043, 0.015, 0.009]
VALUES = [1,2,3,4]
CLOSE_SCALE = 5.0
SLIP_P_TOWARDS = 0.935
# slip_features: edge, close, rss      
















#=================================================USAGE / PIPELINE================================#

def sample_slip(model, features):
    MAX_RANGE = features["nn_norm_range_sep"].max()
    n = features.height

    df_u = features.select(
        edge = (0.5 - pl.col("frac_u").abs() + EDGE_MIN).log(),
        close = ((-pl.col("nn_norm_range_sep").cast(pl.Float64).fill_null(MAX_RANGE))/CLOSE_SCALE).exp(),
        rss = pl.col("rss_max"),
    )
    df_v = features.select(
        edge = (0.5 - pl.col("frac_v").abs() + EDGE_MIN).log(),
        close = ((-pl.col("nn_norm_range_sep").cast(pl.Float64).fill_null(MAX_RANGE))/CLOSE_SCALE).exp(),
        rss = pl.col("rss_max"),
    )

    p_u = model.predict_proba(df_u)[:,1]
    p_v = model.predict_proba(df_v)[:,1]


    # Sampling slip values and direction
    rng = np.random.default_rng()
    draws = pl.DataFrame({
        # Whether slip or not
        "hit_u" : rng.random(n) < p_u,
        "hit_v" : rng.random(n) < p_v,
        # Toward closest edge or not
        "toward_u": rng.random(n) < SLIP_P_TOWARDS,
        "toward_v": rng.random(n) < SLIP_P_TOWARDS,
        # how much to slip (if it did slip, if not just ignore this value)
        "mag_u": rng.choice(VALUES, size=n, p=SLIP_MAG_ODDS),
        "mag_v": rng.choice(VALUES, size=n, p=SLIP_MAG_ODDS),
        # in case exactly in middle: choose one direction at random to snap
        "coin_u": rng.choice([-1, 1], size=n),
        "coin_v": rng.choice([-1, 1], size=n),
    })

    # Which edge to snap to. If in middle, select at random with coin
    edge_sign_u = (
        pl.when(
            pl.col("frac_u") == 0).then(pl.col("coin_u")
        ).otherwise(pl.col("frac_u").sign().cast(pl.Int64))
    )
    edge_sign_v = (
        pl.when(
            pl.col("frac_v") == 0).then(pl.col("coin_v")
        ).otherwise(pl.col("frac_v").sign().cast(pl.Int64))
    )

    # Whether to snap to proper edge, or randomly to incorrect edge
    direction_u = pl.when(
        pl.col("toward_u")
    ).then(edge_sign_u).otherwise(-edge_sign_u)

    direction_v = pl.when(
        pl.col("toward_v")
    ).then(edge_sign_v).otherwise(-edge_sign_v)

    df = (
            pl.concat([features, draws], how = "horizontal")
            .select(
                slip_u = pl.when(
                    pl.col("hit_u")).then(direction_u * pl.col("mag_u")
                ).otherwise(0).cast(pl.Int64),
                slip_v = pl.when(
                    pl.col("hit_v")).then(direction_v * pl.col("mag_v")
                ).otherwise(0).cast(pl.Int64)
            )    
        )
    return df



def angular_error(model, features):
    slips = sample_slip(model, features)

    features = pl.concat([features, slips], how = "horizontal")

    df = features.select(
        pl.col("u_true"),
        pl.col("v_true"),
        pl.col("rx_azimuth_deg"),
        pl.col("rx_elevation_deg"),
        ((NGRID * pl.col("u_true") - pl.col("frac_u") + pl.col("slip_u")) / NGRID).alias("u_est"),
        ((NGRID * pl.col("v_true") - pl.col("frac_v") + pl.col("slip_v")) / NGRID).alias("v_est"),
        #((NGRID * pl.col("u_true") - pl.col("frac_u")) / NGRID).alias("u_est"),
        #((NGRID * pl.col("v_true") - pl.col("frac_v")) / NGRID).alias("v_est"),
    )
    az_est, el_est = inv_uv("u_est", "v_est")
    df = df.with_columns(
            az_est.alias("az_est"),
            el_est.alias("el_est"),
            (pl.col("u_true") - pl.col("u_est")).alias("u_error"),   # = (frac - slip)/NGRID
            (pl.col("v_true") - pl.col("v_est")).alias("v_error"),
        ).with_columns(
            # wrap az error into [-180, 180)
            (((pl.col("az_est") - pl.col("rx_azimuth_deg") + 180) % 360) - 180).alias("az_err"),
            (pl.col("el_est") - pl.col("rx_elevation_deg")).alias("el_err"),
        )
    return df


f = features.drop(["slip_u", "slip_v"])
ret = angular_error(model,f)



#==================================================Plotting======================================#
import plotly.graph_objects as go
from plotly.subplots import make_subplots

a = labels["rx_azimuth_error_deg"]
b = labels["rx_elevation_error_deg"]
c = ret["el_err"]
d = ret["az_err"]

series_dict = {"real az": a, "real el": b, "sim el": c, "sim az": d}

fig = go.Figure()

# Add each Polars series as a trace
for name, series in series_dict.items():
    fig.add_trace(
        go.Histogram(
            x=series.to_numpy(),  # Convert Polars Series to NumPy array
            name=name,
            opacity=0.6,  # Translucency for overlapping view
        )
    )

# Configure layout for overlapping histograms
fig.update_layout(
    barmode="overlay",  # Key parameter to overlay bars instead of stacking/side-by-side
    title="Overlapping Histograms (Polars Series)",
    xaxis_title="Value",
    yaxis_title="Count",
)

# Optional: Add dropdown/toggle controls
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "label": "Show All",
                    "method": "update",
                    "args": [{"visible": [True, True, True, True]}],
                },
                {
                    "label": "Only A & B",
                    "method": "update",
                    "args": [{"visible": [True, True, False, False]}],
                },
                {
                    "label": "Only C & D",
                    "method": "update",
                    "args": [{"visible": [False, False, True, True]}],
                },
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.1,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top",
        }
    ]
)

# Render plot
fig.show()

