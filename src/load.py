import sys
import polars as pl
from pathlib import Path
from src.features import *

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
    return pl.concat([features, labels], how = "horizontal")


def load_indep(path :str = CACHE_PATH, name : str = INDEP_NAME) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = pl.read_csv(path + name)
    labels = df["detected"].to_numpy()
    features = df.drop()
    

#-------------------------------Pairwise Classifier--------------------------------------#




#-------------------------------3D Positional Error--------------------------------------#





#--------------------------------------API / Usage----------------------------------#




#---------------------------Example Usage-------------------------------#
features = pl.read_csv("data/feature_cache/all_labels_features.csv")
indep_features = load_indep_features(features)
indep_features.write_csv("data/feature_cache/indep_features.csv")