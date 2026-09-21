from src.load import *
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

'''
First: computes angular error through:
range
u/v for azimuth + elevation

First get az el -> u v
Also get frac_u, frac_v

Then subtract
u - frac_u
v - frac_v

sample slip_u, slip_v binned by frac (high dependence on these guys

convert u_est, v_est -> az_est, el_est
subtract


Next range error:

ml model to predict the bias (average range error)
Two parts:
    (1) Train on entire data set normally, this will be our "mean error"
    (2) Use 5 fold grouped cross validation and save the residuals. We 
    use these to train standard deviation

PROBLEM!!!
Too many extreme values that are hijacking training, since
close ish extreme predictions > many (relatively bad) bulk predictions 
(Simpson's paradox?)

But looks like downstream, itsfine?

however too noisy so train separate model to find the 
standard deviation

aggregate all z-scores (find distribution of range errors)
randomly sample a z-score to scale the std dev by



'''

tags, labels, features = load_pos()


#------------------------------------------Angular Error--------------------------------------#
# (No sampling for now)
from src.feature_builder import *
NGRID = 32.0
AZ0 = 30.0
'''
sim = features.select(
    pl.col("u_true"),
    pl.col("v_true"),
    ((NGRID * pl.col("u_true") - pl.col("frac_u"))/ NGRID).alias("u_est"),
    ((NGRID * pl.col("v_true") - pl.col("frac_v"))/ NGRID).alias("v_est"),
    pl.col("rx_azimuth_deg"),
    pl.col("rx_elevation_deg")
)


az_est, el_est = inv_uv("u_est", "v_est")
sim = sim.with_columns(
    az_est.alias("az_est"),
    el_est.alias("el_est"),
    (pl.col("u_true") - pl.col("u_est")).alias("u_error"),
    (pl.col("v_true") - pl.col("v_est")).alias("v_error"),
).with_columns(
    (pl.col("az_est") - pl.col("rx_azimuth_deg")).alias("az_err"),
    (pl.col("el_est") - pl.col("rx_elevation_deg")).alias("el_err"),
)
'''




#-----------------------Predicting average error-----------------------#
'''
features = features.drop(
    "u_true",
    "v_true",
    "slip_u",
    "slip_v"
)
'''

# 5 fold grouped cross validation : fitting exactly to error overfits and makes
# the residuals too small

FEATURES = [f.value for f in PosFeature]
NUM_ROUNDS = 500  

range_error = labels["range_error_m"]
DEFAULT_SPLIT =  train_test_split(features, range_error, test_size=0.2, random_state=42)
[X_train, X_val, y_train, y_val] = DEFAULT_SPLIT
GBM_PARAMS = {
    "objective": "regression",
    "metric" : "rmse",
    "n_estimators" : 200,
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


def train_bias(split = DEFAULT_SPLIT, params = GBM_PARAMS):
    [X_train, X_val, y_train, y_val] = split
    feature_names = features.columns
    dtrain = lgb.Dataset(X_train, label=y_train.to_numpy(), feature_name=feature_names)
    dval   = lgb.Dataset(X_val, label=y_val.to_numpy(), reference=dtrain, feature_name=feature_names)


    model = lgb.train(params, dtrain,
                    valid_sets=[dval], valid_names=["val"],
                    callbacks=[lgb.log_evaluation(period = 10)])
    return model


#model = train_bias()
#model.save_model("models/position_bias.txt")
model = lgb.Booster(model_file = "models/position_bias.txt")
from sklearn.metrics import r2_score
y_pred = model.predict(X_val)
r2 = r2_score(y_val, y_pred)

y_val = y_val.to_numpy()

print(f"R^2 Score : {r2}")
print(np.std(y_pred) / np.std(y_val)) 
print(np.corrcoef(y_pred, y_val)[0,1])
print(np.mean(y_pred) - np.mean(y_val))

GROUP_COLS = ["scenario_id", "drop_id"]
N_FOLDS = 5
SEED = 0
# Assigns whole drops to folds, prevents cross drop leakage across train-test

def assign_folds(df : pl.DataFrame, n_folds : int = N_FOLDS, seed : int = SEED):
    folds = (
        df.select(GROUP_COLS)
        .unique()
        .sort(GROUP_COLS)
        .sample(fraction = 1.0, shuffle = True, seed = seed)
        .with_columns((pl.int_range(pl.len()) % n_folds).alias("fold"))
    )
    return df.join(folds, on = GROUP_COLS, how = "left")

def predict_out_of_fold(features : pl.DataFrame, labels : pl.DataFrame, tags : pl.DataFrame, params : dict):
    X = features.to_numpy()
    y = labels.to_numpy()
    fold = tags["fold"].to_numpy()

    out_of_fold = np.full(X.shape[0], np.nan)

    for k in range(N_FOLDS): 
        print(f"Cross fitting fold {k}")
        (train, valid) = (fold!= k, fold == k)  
        dtrain = lgb.Dataset(X[train], label = y[train], feature_name = FEATURES)
        dvalid = lgb.Dataset(X[valid], label = y[valid], feature_name = FEATURES, reference = dtrain)
        model = lgb.train(params, dtrain, 
                          valid_sets = [dvalid],
                          valid_names = ["valid"],
                          callbacks=[lgb.log_evaluation(period=50)])
        out_of_fold[valid] = model.predict(X[valid])
    return out_of_fold



tags = assign_folds(tags)

#print(features.columns)
#print(FEATURES)
#oof = predict_out_of_fold(features, labels["range_error_m"], tags, GBM_PARAMS)
#np.savetxt("data/feature_cache/oof_outputs.csv", oof, delimiter = ",")
oof_predictions = np.genfromtxt("data/feature_cache/oof_outputs.csv")
real_errors = labels.select("range_error_m")
real_errors.write_csv("data/feature_cache/real_range_error.csv")


#----------------------Training on standard deviation--------------------#
residuals = labels.select(
    ((pl.col("range_error_m") - oof_predictions)).alias("residual_squared")
)

import plotly.graph_objects as go
df = residuals.select(pl.col("residual_squared"))

fig = go.Figure(
    go.Histogram(x=df["residual_squared"], nbinsx=100)
)
fig.update_layout(
    xaxis_title="residual_squared",
    yaxis_title="count",
    bargap=0.1,          # px adds a small gap between bars; drop for touching bars
)
fig.show()

'''
print(real_errors.to_numpy().ravel())
fig1 = go.Figure()
bins = dict(start=-100, end=100, size=0.04)
fig1.add_trace(go.Histogram(x=y_pred, name="model predictions"))
fig1.add_trace(go.Histogram(x=y_val, name="true error"))

fig1.update_layout(
    barmode="overlay",   
    xaxis_title="range error",
    yaxis_title="count",
    bargap=0.1,
)
fig1.update_traces(opacity=0.6)   
fig1.show()
'''