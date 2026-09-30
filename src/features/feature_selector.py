from src.data.features import IndepFeature, PosFeature, PairFeature
import polars as pl


MATCH_COLS = [
    "scenario_id",
    "drop_id",
    "nearest_neighbor_xyz_distance_m",
    "nearest_neighbor_bistatic_range_sep_m",
    "nearest_neighbor_rx_az_sep_deg",
    "nearest_neighbor_rx_el_sep_deg",
    "nearest_neighbor_tx_az_sep_deg",
    "nearest_neighbor_tx_el_sep_deg",
    "nearest_neighbor_norm_range_sep",
    "nearest_neighbor_norm_rx_angle_sep",
]



def indep_mutual_split(df : pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    df_tagged = df.with_columns(
        group_size = pl.len().over(MATCH_COLS)
    )

    mutual_pairs = df_tagged.filter(pl.col("group_size") == 2).drop("group_size")
    non_mutual = df_tagged.filter(pl.col("group_size") == 1).drop("group_size")

    return mutual_pairs, non_mutual

def process_indep_features(df : pl.DataFrame) -> pl.DataFrame:
    (_ , indep) = indep_mutual_split(df)
    feature_list = [f.value for f in IndepFeature]
    labels = indep.select("detected")
    features = indep.select(feature_list)
    tags = indep.select(["scenario_id", "drop_id", "target_id"])
    return pl.concat([tags, features, labels], how = "horizontal")



#=====================================================Pair Features==============================================#
SHARED_FEATURES = [
    PosFeature.N_TARGETS.value,
    PosFeature.NN_NORM_RANGE_SEP.value,
    PosFeature.NN_NORM_ANG_SEP.value
]

FEAT = [feat.value for feat in PairFeature if feat not in SHARED_FEATURES]
FEATURE_COLS = [f"{f}_{suffix}" for f in FEAT for suffix in ("a", "b")]
def process_pair_features(df: pl.DataFrame) -> pl.DataFrame:
    pair, _ = indep_mutual_split(df)

    shared_tags = ["scenario_id", "drop_id"]
    per_target = ["target_id", "detected", *FEAT]
    left_keys = list(dict.fromkeys(MATCH_COLS + shared_tags))  # avoid dup cols

    a = pair.select(
        *left_keys,
        pl.col(per_target).name.suffix("_a"),
        *SHARED_FEATURES,  # taken from the "a" row, as in your loop
    )
    b = pair.select(
        *MATCH_COLS,
        pl.col(per_target).name.suffix("_b"),
    )

    cols = (shared_tags + ["target_id_a", "target_id_b"] + FEATURE_COLS
            + SHARED_FEATURES + ["detected_a", "detected_b"])

    return (
        a.join(b, on=MATCH_COLS)
         .filter(pl.col("target_id_a") != pl.col("target_id_b"))
         .select(cols)
    )


def drop_missed(df : pl.DataFrame) -> pl.DataFrame:
    filtered = df.filter(
        (pl.col("detected") == 1)
    )
    return filtered

def process_pos_features(df : pl.DataFrame) -> pl.DataFrame:
    cleaned = drop_missed(df)
    feature_list = [f.value for f in PosFeature]
    features = cleaned.select(feature_list)

    labels = cleaned.select(["position_error_3d_m","range_error_m", "rx_azimuth_error_deg","rx_elevation_error_deg","slip_v","slip_u"])


    tags = cleaned.select(
        ["scenario_id", "drop_id", "target_id"]
    )

    return pl.concat([tags, features, labels], how = "horizontal")