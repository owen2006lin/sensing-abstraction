from src.load import *
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
                    )


slip_label_v = sim.select(pl.col("slip_v").alias("slip"))
slip_features_v = (features.
                            select(["rss_max", "nn_norm_range_sep","abs_frac_v"])
                            .rename({"abs_frac_v" : "abs_frac"})
                    )

slip_labels = slip_label_u.vstack(slip_label_v)
slip_features = slip_features_u.vstack(slip_features_v)

print(slip_labels)
print(slip_features)    






