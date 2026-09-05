from src.load import *
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupShuffleSplit

tags, labels, features = load_indep()

GBM_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
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

# Splits by (scenario_id, drop_id) so targets from the same drop can't land on both
# sides of the split (they share TX/RX geometry and channel/clutter realization).
GROUP_SHUFFLE_SPLIT = load_group_split()

# Split randomly by default, but there may be leakage with instances cross the train test boundary
DEFAULT_SPLIT =  load_default_split()

#Actions to do during training
CALLBACKS = [
    # stops after 200 rounds of no improvement
    lgb.early_stopping(stopping_rounds = 200), 
    lgb.log_evaluation(period = 2)]

def train_indep(split = DEFAULT_SPLIT, params = GBM_PARAMS, n_rounds : int = 5000, callbacks = CALLBACKS) -> lgb.Booster:
    [X_train, X_val, y_train, y_val] = split
    feature_names = features.columns
    dtrain = lgb.Dataset(X_train, label=y_train.to_numpy(), feature_name=feature_names)
    dval   = lgb.Dataset(X_val, label=y_val.to_numpy(), reference=dtrain, feature_name=feature_names)


    model = lgb.train(params, dtrain, num_boost_round=n_rounds,
                    valid_sets=[dval], valid_names=["val"],
                    callbacks=callbacks)
    return model



#------------------------------USAGE-------------------#
model = train_indep()
model.save_model("models/indep_model.txt")

model_gss = train_indep(split = GROUP_SHUFFLE_SPLIT)
model_gss.save_model("models/indep_model_gss.txt")

#loading
model = lgb.Booster(model_file = "models/indep_model.txt")

