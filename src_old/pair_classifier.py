from src.load import *
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split

tags, labels, features = load_pairs()
DEFAULT_SPLIT =  train_test_split(features, labels, test_size=0.2, random_state=42)

GBM_PARAMS = {
    "objective": "multiclass",
    "num_class" : 4,
    "metric": "multi_logloss",
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
CALLBACKS = [
    # stops after 200 rounds of no improvement
    lgb.early_stopping(stopping_rounds = 200), 
    lgb.log_evaluation(period = 2)]


def train_pair(split = DEFAULT_SPLIT, params = GBM_PARAMS, n_rounds : int = 5000, callbacks = CALLBACKS) -> lgb.Booster:
    [X_train, X_val, y_train, y_val] = split
    feature_names = features.columns
    dtrain = lgb.Dataset(X_train, label=y_train.to_numpy(), feature_name=feature_names)
    dval   = lgb.Dataset(X_val, label=y_val.to_numpy(), reference=dtrain, feature_name=feature_names)


    model = lgb.train(params, dtrain, num_boost_round=n_rounds,
                    valid_sets=[dval], valid_names=["val"],
                    callbacks=callbacks)
    return model


model_pair = train_pair()
model_pair.save_model("models/pair_model.txt")