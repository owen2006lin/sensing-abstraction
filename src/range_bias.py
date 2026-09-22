from src.load import *
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


tags, labels, features = load_pos()
range_error = labels["range_error_m"]

#Grouped val split for hyperparameter tuning
groups = tags.select(pl.struct(["scenario_id", "drop_id"]).hash()).to_series().to_numpy()
gss = GroupShuffleSplit(n_splits = 1, test_size = 0.15, random_state = 42)
train_idx, val_idx = next(gss.split(features, labels, groups = groups))


[X_val, y_val, tags_val] = [features[val_idx], range_error[val_idx], tags[val_idx]]
[X, y, tags] = [features[train_idx], range_error[train_idx], tags[train_idx]]


#-------------------------------------Residuals with OOF-----------------------------#
GBM_PARAMS = {
    "objective": "regression",
    "metric" : "rmse",
    "n_estimators" : 3000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.65,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,

    #Use cuda to speed up (can ignore if on CPU)
    'device': 'cuda',       
    'gpu_use_dp': False,
}
FEATURES = [f.value for f in PosFeature]
GROUP_COLS = ["scenario_id", "drop_id"]
N_FOLDS = 5
SEED = 0

def assign_folds(df : pl.DataFrame, n_folds : int = N_FOLDS, seed : int = SEED):
    folds = (
        df.select(GROUP_COLS)
        .unique()
        .sort(GROUP_COLS)
        .sample(fraction = 1.0, shuffle = True, seed = seed)
        .with_columns((pl.int_range(pl.len()) % n_folds).alias("fold"))
    )
    return df.join(folds, on = GROUP_COLS, how = "left", maintain_order = "left")

def predict_out_of_fold(features : pl.DataFrame, labels : pl.DataFrame, tags : pl.DataFrame, params : dict):
    X = features.to_numpy()
    y = labels.to_numpy()
    fold = tags["fold"].to_numpy()
    models = []
    out_of_fold = np.full(X.shape[0], np.nan)
    for k in range(N_FOLDS): 
        print(f"Cross fitting fold {k}")
        (train, valid) = (fold!= k, fold == k)  
        dtrain = lgb.Dataset(X[train], label = y[train], feature_name = FEATURES)
        dvalid = lgb.Dataset(X_val, label = y_val, feature_name = FEATURES, reference = dtrain)
        model = lgb.train(params, dtrain, 
                          valid_sets = [dvalid],
                          valid_names = ["valid"],
                          callbacks=[lgb.log_evaluation(period=50),
                                     lgb.early_stopping(stopping_rounds=100, verbose=True)])
        print(model.best_iteration)
        out_of_fold[valid] = model.predict(X[valid])
        models.append(model)

    return out_of_fold, models

def predict_ensemble(models, X_val):
    preds = np.column_stack([
        m.predict(X_val, num_iteration = m.best_iteration) for m in models
    ])
    return preds.mean(axis = 1)

def train_write_oof(X,y, tags, params):
    tags = assign_folds(tags)
    oof, models = predict_out_of_fold(X,y,tags, params)

    residuals = y - oof 
    residuals_val = y_val - predict_ensemble(models, X_val)

    np.savetxt("data/feature_cache/residuals.csv", residuals, delimiter = ",")
    np.savetxt("data/feature_cache/residuals_val.csv", residuals_val, delimiter = ",")

train_write_oof(X,y,tags, GBM_PARAMS)

#==============================Training on raw bias==============================#
DEFAULT_SPLIT =  train_test_split(X, y, test_size=0.2, random_state=42)
def train_bias(split = DEFAULT_SPLIT, params = GBM_PARAMS):
    [X_train, X_test, y_train, y_test] = split
    feature_names = features.columns
    dtrain = lgb.Dataset(X_train, label=y_train.to_numpy(), feature_name=feature_names)
    dvalid = lgb.Dataset(X_val, label = y_val, feature_name = FEATURES, reference = dtrain)
    model = lgb.train(params, dtrain,
                    valid_sets=[dvalid], valid_names=["val"],
                    callbacks=[lgb.log_evaluation(period = 10),
                               lgb.early_stopping(stopping_rounds=100, verbose=True)])
    return model

#model = train_bias()
#model.save_model("models/position_bias.txt")
model = lgb.Booster(model_file = "models/position_bias.txt")
preds = model.predict(X)

'''
# Plotting some visualizations
y_arr = np.asarray(y).ravel()
print(y_arr.std())
print(np.sqrt(np.mean((oof_predictions-y_arr)**2)))


import plotly.graph_objects as go
residuals = y - oof_predictions

def plot(data, name):
    fig = go.Figure(
    go.Histogram(x=data, nbinsx=100)
    )
    fig.update_layout(
        xaxis_title=name,
        yaxis_title="count",
        bargap=0.1,          # px adds a small gap between bars; drop for touching bars
    )
    fig.show()

#plot(y, "actual")
#plot(oof_predictions, "oof predictions")
#plot(residuals, "residuals")
#plot(preds, "bias model")
'''
