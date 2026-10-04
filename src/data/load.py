import sys
import polars as pl
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from src.data.features import IndepFeature, PairFeature, PosFeature
from sklearn.model_selection import train_test_split
from src.feature_building.snr import build_snr_features

CACHE_PATH = "data/feature_cache/"
INDEP_NAME = "indep_features.csv"
PAIR_NAME = "pair_features.csv"
ERROR_NAME = "error_features.csv"
POS_NAME = "pos_features.csv"
CLASS_NAME = "full_class_features.csv"

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

PATH = "data/raw"
csvs = ["target", "detection", "error", "snr"]
KEYS = ['scenario_id', 'drop_id', 'target_id']

def load_raw(path : str = PATH, names : list[str] = csvs) -> list[pl.DataFrame]:
    dfs = []
    for name in names:
        df = pl.read_csv(f"{path}/{name}.csv")
        dfs.append(df.with_columns(pl.col(pl.Float64, pl.Float32).fill_nan(None)))
    return dfs

# For now, targets don't have an explicit detected/not label which is super annoying
def label_error(error : pl.DataFrame) -> pl.DataFrame:
    error = error.with_columns(
        pl.col("position_error_x_m").is_not_null().cast(pl.Int8).alias("detected")
    )
    return error

def combine(target: pl.DataFrame, detect : pl.DataFrame, error: pl.DataFrame, snr: pl.DataFrame):
    error = label_error(error)
    snr = build_snr_features(snr)
    combined = target.join(error, on = KEYS, how = "left").join(snr, on = KEYS, how = "left")

    return combined

def group_split(tags, labels, features, return_tags = False):
    groups = tags.select(pl.struct(["scenario_id", "drop_id"]).hash()).to_series().to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(features, labels, groups=groups))
    gss_split = [features[train_idx], features[val_idx], labels[train_idx], labels[val_idx]]
    if return_tags:
        gss_split = [features[train_idx], features[val_idx], labels[train_idx], labels[val_idx], tags[train_idx], tags[val_idx]]
    return gss_split

    
def tt_split(labels, features, tags = None):
    [X_train, X_val, y_train, y_val] =  train_test_split(features, labels, test_size=0.2, random_state=42)
    return [X_train, X_val, y_train, y_val]
#--------------------------------Independent Classifier---------------------------------#


def load_indep(path :str = CACHE_PATH, name : str = INDEP_NAME) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    df = pl.read_csv(path + name, infer_schema_length=None)
    labels = df.select("detected")

    feature_list = [f.value for f in IndepFeature]
    features = df.select(feature_list)
    tags = df.select(
        ["scenario_id", "drop_id", "target_id"]
    )
    return tags, labels, features


SHARED_FEATURES = [
    PairFeature.N_TARGETS.value,
    PairFeature.NN_NORM_RANGE_SEP.value,
    PairFeature.NN_NORM_ANG_SEP.value
]
FEAT = [feat.value for feat in PairFeature if feat not in SHARED_FEATURES]
FEATURE_COLS = [f"{f}_{suffix}" for f in FEAT for suffix in ("a", "b")]

def load_pairs(path :str = CACHE_PATH, name : str = PAIR_NAME):
    df = pl.read_csv(path + name, infer_schema_length=None)
    labels = df.select(["detected_a", "detected_b"])
    labels = labels.select(
        (2 * (pl.col("detected_a")) + (pl.col("detected_b"))).alias("score")
    )

    feature_list = FEATURE_COLS + SHARED_FEATURES
    features = df.select(feature_list)
    tags = df.select(
        ["scenario_id", "drop_id", "target_id_a", "target_id_b"]
    )
    return tags, labels, features

#-------------------------------3D Positional Error--------------------------------------#


def load_pos(path : str = CACHE_PATH, name : str = POS_NAME):
    df = pl.read_csv(path + name, infer_schema_length=None)

    labels = df.select(["position_error_3d_m","range_error_m", "rx_azimuth_error_deg","rx_elevation_error_deg","slip_v","slip_u"])

    feature_list = [f.value for f in PosFeature]
    features = df.select(feature_list)

    tags = df.select(
        ["scenario_id", "drop_id", "target_id"]
    )

    return tags, labels, features

#======================================Full Classifier===========================================#
# Loading for full classifier pipeline
def load_class(path : str = CACHE_PATH, name : str = CLASS_NAME):
    df = pl.read_csv(path + name, infer_schema_length=None)
    
    labels = df.select("detected")
    features_indep = [f.value for f in IndepFeature]
    features_pair = [f.value for f in PairFeature]
    feature_list = list(set(features_indep) | set(features_pair))

    features = df.select(feature_list)
    tags = df.select(
        ["scenario_id", "drop_id", "target_id"]
    )
    return tags, labels, features


def split_indep(df, feature_list):
    tags = df.select(["scenario_id", "drop_id", "target_id"])
    labels = df.select("detected")
    features = df.select(feature_list)

    return tags, labels, features

def split_pairs(df, feature_list):
    tags = df.select(
        ["scenario_id", "drop_id", "target_id_a", "target_id_b"]
    )
    labels =  (df
                .select(["detected_a", "detected_b"])
                .select((2 * (pl.col("detected_a")) + (pl.col("detected_b")))
                .alias("score")
                ))
    features = df.select(feature_list)  

    return tags, labels, features
