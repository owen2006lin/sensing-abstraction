from src.utils.plotting import plot_hist
from src.components.position import FullPositionModel
from src.data.load import load_pos, group_split

import polars as pl
import numpy as np

tags, labels, features = load_pos()
[X, X_val, y, y_val, t, t_val] = group_split(tags, labels, features, return_tags=True)

train_df = pl.concat([X, y, t], how = "horizontal")
val_df = pl.concat([X_val, y_val, t_val], how = "horizontal")

rng = np.random.default_rng(seed = 42)
pos_model = FullPositionModel.load("models/position_model.joblib")
df = pos_model.sample(X_val, rng)
df_train = pos_model.sample(X, rng)

pred_err_val = df["err_3d"].to_numpy().ravel()
err_val = y_val["position_error_3d_m"].to_numpy().ravel()
pred_err_train = df_train["err_3d"].to_numpy().ravel()
err_train = y["position_error_3d_m"].to_numpy().ravel()

names = ["train 3d error", "train (predicted) 3d error", "val 3d error", "val (predicted) 3d error"]
plot_hist([err_train, pred_err_train, err_val, pred_err_val], names)