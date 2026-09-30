from src.load import *
from sklearn.linear_model import LogisticRegression
from src.feature_builder import *
import numpy as np

tags, labels, features = load_pos()

NGRID = 32.0
AZ0 = 30.0
EDGE_MIN = 1e-3  
SLIP_MAG_ODDS = [0.933, 0.043, 0.015, 0.009]
VALUES = [1,2,3,4]
CLOSE_SCALE = 5.0
SLIP_P_TOWARDS = 0.935



# Pad out nn norm range sep NaN, take to max value to preserve scaling
# Also builds edge and close
def build_edge_close(features):
    MAX = features["nn_norm_range_sep"].max()
    features = features.with_columns(
        pl.col("nn_norm_range_sep").fill_null(MAX)
    )

    features = features.with_columns(
        (0.5 - pl.col("abs_frac") + EDGE_MIN).log().alias("edge"),
        (pl.col("nn_norm_range_sep") / (-5)).exp().alias("close"),
        (pl.col("rss_max")).alias("rss")
    ).select(
        ["edge", "close", "rss"]
    )

    return features

def build_slip_features(features, labels):
    df = features.select(
        pl.col("u_true"),
        pl.col("v_true"),
        ((NGRID * pl.col("u_true") - pl.col("frac_u"))/ NGRID).alias("u_est"),
        ((NGRID * pl.col("v_true") - pl.col("frac_v"))/ NGRID).alias("v_est"),
        pl.col("rx_azimuth_deg"),
        pl.col("rx_elevation_deg"),
    )

    slips = labels.select(
        pl.col("slip_v"),
        pl.col("slip_u")
    )

    slip_label_u = slips.select(pl.col("slip_u").alias("slip"))
    slip_features_u = (features.select(["rss_max", "nn_norm_range_sep", "abs_frac_u"]) 
                            .rename({"abs_frac_u" : "abs_frac"})
                            .with_columns(
                                pl.lit("u").alias("type")
                            )
                    )

    slip_label_v = slips.select(pl.col("slip_v").alias("slip"))
    slip_features_v = (features.
                                select(["rss_max", "nn_norm_range_sep","abs_frac_v"])
                                .rename({"abs_frac_v" : "abs_frac"})
                                .with_columns(
                                    pl.lit("v").alias("type")
                                )
                        )

    slip_labels = slip_label_u.vstack(slip_label_v)
    slip_labels = slip_labels.select(
        (pl.col("slip") != 0).alias("slipped")
    ).select("slipped")

    slip_features = slip_features_u.vstack(slip_features_v)
    slip_tags = slip_features.select("type")
    slip_features = slip_features.drop("type")

    return slip_labels, slip_tags, slip_features



def train_slip_model(slip_features : pl.DataFrame, slip_labels : pl.Series, print_params = False) -> LogisticRegression:
    model = LogisticRegression(C = 10, max_iter = 1000)
    model.fit(slip_features, slip_labels)
    if print_params:
        b0 = model.intercept_[0]
        b1,b2,b3 = model.coef_[0]
        for name, c in zip(slip_features.columns, [b1,b2,b3]):
            print(f"{name}={c:.3f}")
        print(f"intercept={b0:.3f}")
    return model


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
