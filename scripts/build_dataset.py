import sys
import polars as pl
from src.feature_building.nn import build_nn_features
from src.feature_building.geo import build_geometry_features
from src.feature_building.quantization import build_angle_features
from src.feature_building.snr import build_snr_features
from src.feature_building.feature_selector import process_indep_features, process_pair_features, process_pos_features, process_classifier_features


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


#---------------------------------------------------------------Example Usage----------------------------------------------------------------#
# Building all possible features

[target, detect, error, snr] = load_raw()
error = label_error(error)
snr_features = build_snr_features(snr)
nn_features = build_nn_features(target)


# some changes to detection before merging, changing column name "associated_target_id" -> "target_id"
# and also dropping false positives for now
detect = detect.rename({"associated_target_id" : "target_id"})
detect = detect.drop_nans("target_id").with_columns(
    pl.col("target_id").cast(pl.Int64)
)
combined = target.join(detect, on = KEYS, how = "left")
angle_features = build_angle_features(combined)
geo_features = build_geometry_features(combined)
# Completed dataframe with all features and labels, to cache
merged_raw = (target
                .join(detect, on = KEYS, how = "left")
                .join(error, on = KEYS, how = "left")
                .join(snr_features, on = KEYS, how = "left"))

merged = pl.concat([merged_raw, geo_features, angle_features, nn_features], how = "horizontal")
merged.write_csv("data/feature_cache/all_labels_features.csv")




# For selecting / pre loading individual model features

features = pl.read_csv("data/feature_cache/all_labels_features.csv")
indep_features = process_indep_features(features)
indep_features.write_csv("data/feature_cache/indep_features.csv")

pairs = process_pair_features(features)
pairs.write_csv("data/feature_cache/pair_features.csv")

pos = process_pos_features(features)
pos.write_csv("data/feature_cache/pos_features.csv")

# Loading / caching full classifier features

features = pl.read_csv("data/feature_cache/all_labels_features.csv")

class_features = process_classifier_features(features)
class_features.write_csv("data/feature_cache/full_class_features.csv")
