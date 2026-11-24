import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from src.data.load import load_class, load_pos, group_split
from src.components.detection import FullClassifier
from src.components.position import FullPositionModel

# Fits every production model from the cached features (data/feature_cache, built by scripts/build_dataset.py)
# and saves them to models/. Run from the repo root : python scripts/train.py
MODEL_DIR = "models/"


# Outer split holds out a test set for evaluation, inner split carves an early stopping val set from train
def split(tags, labels, features):
    [X, X_test, y, y_test, t, t_test] = group_split(tags, labels, features, return_tags = True)
    [X, X_val, y, y_val, t, t_val] = group_split(t, y, X, return_tags = True)
    train_df = pl.concat([X, y, t], how = "horizontal")
    val_df = pl.concat([X_val, y_val, t_val], how = "horizontal")
    return train_df, val_df


#----------------------------------------------Detection----------------------------------------------#
# FullClassifier trains the indep and pair classifiers internally, so save those members directly
train_df, val_df = split(*load_class())
full_classifier = FullClassifier().fit(train_df, val_df)
full_classifier.save(MODEL_DIR + "full_classifier.joblib")
full_classifier.indep_model.save(MODEL_DIR + "indep_classifier.joblib")
full_classifier.pair_model.save(MODEL_DIR + "pair_classifier.joblib")


#----------------------------------------------Position-----------------------------------------------#
# FullPositionModel trains the angle and range models internally, so save the range member directly
train_df, val_df = split(*load_pos())
pos_model = FullPositionModel().fit(train_df, val_df)
pos_model.save(MODEL_DIR + "position_model.joblib")
pos_model.range_model.save(MODEL_DIR + "range_model.joblib")
