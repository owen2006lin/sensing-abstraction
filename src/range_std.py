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


residuals = np.genfromtxt("data/feature_cache/residuals.csv")
std_labels = np.log(residuals**2 + 1e-6)

residuals_val = np.genfromtxt("data/feature_cache/residuals_val.csv")
std_val_labels = np.log(residuals_val ** 2 + 1e-6)

DEFAULT_SPLIT =  train_test_split(X, std_labels, test_size=0.2, random_state=42)
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


def train_std(split = DEFAULT_SPLIT, params = GBM_PARAMS):
    [X_train, X_test, y_train, y_test] = split
    feature_names = features.columns
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dvalid = lgb.Dataset(X_val, label = y_val, feature_name = FEATURES, reference = dtrain)
    model = lgb.train(params, dtrain,
                    valid_sets=[dvalid], valid_names=["val"],
                    callbacks=[lgb.log_evaluation(period = 10),
                               lgb.early_stopping(stopping_rounds=100, verbose=True)])
    return model


#model = train_std()
#model.save_model("models/position_std.txt")
model = lgb.Booster(model_file="models/position_std.txt")
raw_sigma = np.sqrt(np.exp(model.predict(X_val)))

'''
Architecture Decision:

Around 96% of rows is uniformly distributed of flat error +- 0.61m. (Bulk) 
Other 4% of rows is large errors, up to -13.6/+15.4 m, liklihood strongly depends on features (Blunders)

Instead of predicting a single sigma per row, model the two parts separately


Training:
(1) Label each residual |r| > 0.7m a blunder
(2) Store bulk distribution shape
(3) Store blunder distribution shape
(3) Train a LGBM classifier on probability is a blunder given features x

Inference Time:
(1) Predict mu = bias(x), Pi = classifier(x)
(2) Error distribution for single sample is (1 - pi) x bulk + pi x blunder
(3) Draw U. If U < Pi, draw from blunder pool. Otherwise, draw from bulk and add mu
'''



















