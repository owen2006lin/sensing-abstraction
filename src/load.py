import sys
import polars as pl
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from src.features import *
from sklearn.model_selection import train_test_split


CACHE_PATH = "data/feature_cache/"
INDEP_NAME = "indep_features.csv"
PAIR_NAME = "pair_features.csv"
ERROR_NAME = "error_features.csv"

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
    

#--------------------------------Independent Classifier---------------------------------#
def load_indep_features(df : pl.DataFrame) -> pl.DataFrame:
    (_ , indep) = indep_mutual_split(df)
    feature_list = [f.value for f in IndepFeature]
    labels = indep.select("detected")
    features = indep.select(feature_list)
    tags = indep.select(["scenario_id", "drop_id", "target_id"])
    return pl.concat([tags, features, labels], how = "horizontal")


def load_indep(path :str = CACHE_PATH, name : str = INDEP_NAME) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = pl.read_csv(path + name, infer_schema_length=None)
    labels = df.select("detected")

    feature_list = [f.value for f in IndepFeature]
    features = df.select(feature_list)
    tags = df.select(
        ["scenario_id", "drop_id", "target_id"]
    )
    return tags, labels, features


def load_default_split():
    _ , labels, features = load_indep()
    [X_train, X_val, y_train, y_val] =  train_test_split(features, labels, test_size=0.2, random_state=42)
    return [X_train, X_val, y_train, y_val]

# Prefer to use gss to prevent same drop train-validation leakage
def load_group_split():
    tags, labels, features = load_indep()
    groups = tags.select(pl.struct(["scenario_id", "drop_id"]).hash()).to_series().to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(features, labels, groups=groups))
    gss_split = [features[train_idx], features[val_idx], labels[train_idx], labels[val_idx]]
    return gss_split

#-------------------------------Pairwise Classifier--------------------------------------#

# Converts all pairs into single row of features
# Duplicates pairs to enforce symmetry by rearranging order
SHARED_FEATURES = [
    PosFeature.N_TARGETS.value,
    PosFeature.NN_NORM_RANGE_SEP.value,
    PosFeature.NN_NORM_ANG_SEP.value
]

FEAT = [feat.value for feat in PairFeature if feat not in SHARED_FEATURES]
FEATURE_COLS = [f"{f}_{suffix}" for f in FEAT for suffix in ("a", "b")]

def process_pair(df : pl.DataFrame):
    (pair , _) = indep_mutual_split(df)
    grouped = pair.group_by(MATCH_COLS)    

    shared_tags = ["scenario_id", "drop_id"]
    target_ids = ["target_id_a", "target_id_b"]
    features_ab = FEATURE_COLS
    pair_feats = SHARED_FEATURES
    detected_cols = ["detected_a", "detected_b"]
    cols = shared_tags + target_ids + features_ab + pair_feats + detected_cols

    chunks = []

    for key, sub in grouped:
        row_a, row_b = sub[0], sub[1]
        merged = pl.concat(
            [
                row_a.select(shared_tags),
                row_a.select((pl.col("target_id")).name.suffix("_a")),
                row_b.select((pl.col("target_id")).name.suffix("_b")),
                row_a.select(pl.col(FEAT).name.suffix("_a")),
                row_b.select(pl.col(FEAT).name.suffix("_b")),
                row_a.select((pl.col("detected")).name.suffix("_a")),
                row_b.select((pl.col("detected")).name.suffix("_b")),
                row_a.select(SHARED_FEATURES),
            ],
            how = "horizontal"
        )
        chunks.append(merged)

    for key, sub in grouped:
        row_a, row_b = sub[1], sub[0]
        merged = pl.concat(
            [
                row_a.select(shared_tags),
                row_a.select((pl.col("target_id")).name.suffix("_a")),
                row_b.select((pl.col("target_id")).name.suffix("_b")),
                row_a.select(pl.col(FEAT).name.suffix("_a")),
                row_b.select(pl.col(FEAT).name.suffix("_b")),
                row_a.select((pl.col("detected")).name.suffix("_a")),
                row_b.select((pl.col("detected")).name.suffix("_b")),
                row_a.select(SHARED_FEATURES),
            ],
            how = "horizontal"
        )
        chunks.append(merged)

    pairs = pl.concat(chunks, how = "vertical").select(cols)
    return pairs

def load_pairs(path :str = CACHE_PATH, name : str = PAIR_NAME):
    df = pl.read_csv(path + name, infer_schema_length=None)
    labels = df.select(["detected_a", "detected_b"])
    labels = labels.select(
        (2 * (pl.col("detected_a")) + (pl.col("detected_b"))).alias("score")
    )

    feature_list = FEATURE_COLS
    features = df.select(feature_list)
    tags = df.select(
        ["scenario_id", "drop_id", "target_id_a", "target_id_b"]
    )
    return tags, labels, features

#-------------------------------3D Positional Error--------------------------------------#





#--------------------------------------API / Usage----------------------------------#




#---------------------------Example Usage-------------------------------#
features = pl.read_csv("data/feature_cache/all_labels_features.csv")
#indep_features = load_indep_features(features)
#indep_features.write_csv("data/feature_cache/indep_features.csv")

#pairs = process_pair(features)
#pairs.write_csv("data/feature_cache/pair_features.csv")

tags, labels, features = load_pairs()