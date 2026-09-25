from src.load import *
import polars as pl
import numpy as np
import lightgbm as lgb
import plotly.express as px
from sklearn.model_selection import train_test_split
import random


tags, errors, features = load_pos()
range_error = errors.select("range_error_m")

'''
Labeling : 6 classes, bulk, sb+/-, far+/-, near

Thresholds are below:
(1)bulk        :           -0.52m - +0.82m
(2)sideband +  :           1.4m - 1.85m
(3)sideband -  :           -2.25m - -1.55m
(4)far +       :           >  1.85m 
(5)far -       :           < -2.25m
(6)near        :           anything else

'''
c = pl.col("range_error_m")

labels = range_error.with_columns(
    pl.when(c.is_between(-0.52, 0.82)).then(0)
    .when(c.is_between(1.4,1.85)).then(1)
    .when(c.is_between(-2.25, -1.55)).then(2)
    .when(c > 1.85).then(3)
    .when(c < -2.25).then(4)
    .otherwise(5)
    .alias("region")
).select("region")




#=================================Regime Classifier============================#
# Severe problems with memorization and miscalibration: instead we'll use 5 fold cross
# validation
from scipy.optimize import minimize
from scipy.special import softmax

GBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 6,                 # bulk, sb+, sb-, far+, near, far-  (encode labels as 0..5)
    "metric": "multi_logloss",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 4,                 # effectively caps trees at 16 leaves
    "min_data_in_leaf": 200,
    "feature_fraction": 1.0,        # no column subsampling
    "bagging_fraction": 1.0,        # no row subsampling
    "bagging_freq": 0,
    "lambda_l2": 1.0,
    "max_bin": 255,
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

# Fit bias parameter b_k to fix systemic error in classifier
# p_k = exp(z_k) / Σ_j exp(z_j)                 by default
# p_k = exp(z_k + b_k) / Σ_j exp(z_j + b_j)     including bias b_k
def fit_bias(raw, y):                      # raw: (n, 6) held-out raw scores, y: labels 0..5
    def nll(b):
        q = softmax(raw + np.r_[0.0, b], axis=1)
        return -np.log(q[np.arange(len(y)), y] + 1e-12).mean()
    return np.r_[0.0, minimize(nll, np.zeros(raw.shape[1] - 1), method="L-BFGS-B").x]

def apply_bias(raw, b):
    return softmax(raw + b, axis=1)

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
    tags = assign_folds(tags)
    fold = tags["fold"].to_numpy()
    models = []
    n_class = params.get("num_class", 1)
    out_of_fold = np.full((X.shape[0], n_class), np.nan)

    for k in range(N_FOLDS): 
        print(f"Cross fitting fold {k}")
        (train, valid) = (fold!= k, fold == k)  
        dtrain = lgb.Dataset(X[train], label = y[train], feature_name = FEATURES)
        dvalid = lgb.Dataset(X[valid], label = y[valid], feature_name = FEATURES, reference = dtrain)
        model = lgb.train(params, dtrain, 
                          valid_sets = [dvalid],
                          valid_names = ["valid"],
                          callbacks=[lgb.log_evaluation(period=50),
                                     lgb.early_stopping(stopping_rounds=100, verbose=True)])
        print(model.best_iteration)
        out_of_fold[valid] = model.predict(X[valid], raw_score = True)
        models.append(model)

    #==============Cross fit bias, fit on 4 folds then apply to 5th==============#
    y_int = y.ravel().astype(int)
    oof_cal = np.full_like(out_of_fold, np.nan)
    biases = []
    for k in range(N_FOLDS):
        (train, valid) = (fold != k, fold == k)
        b_k = fit_bias(out_of_fold[train], y_int[train])
        oof_cal[valid] = apply_bias(out_of_fold[valid], b_k)
        biases.append(b_k)
        print(f"fold {k} bias: {np.round(b_k, 3)}")

    b_final = fit_bias(out_of_fold, y_int)    #fit on all OOF rows
    return out_of_fold, oof_cal, b_final, models

[X_train, X_val, y_train, y_val] =  train_test_split(features, labels, test_size=0.2, random_state=42)
[tags_train, tags_val] = train_test_split(tags, test_size = 0.2, random_state= 42)

oof_preds, oof_cal, b_final, models = predict_out_of_fold(X_train, y_train, tags_train, GBM_PARAMS)
#np.savetxt("data/feature_cache/oof_class_preds.csv", oof_cal)
print(b_final)

i = 0
for model in models:
    model.save_model(f"models/regime_class_{i}.txt")
    i+=1

#Turns out, b_final = [ 0.          0.17261316  0.1903196  -0.0079811   0.4821967   0.09235376]
#b_final = [ 0. ,0.17261316,  0.1903196,  -0.0079811,   0.4821967,   0.09235376]

def predict_regime(features):
    probs_outputs = 0
    for i in range(N_FOLDS):
        model = lgb.Booster(model_file=f"models/regime_class_{i}.txt")
        preds = model.predict(features, raw_score = True)
        prob = apply_bias(preds, b_final)
        probs_outputs += prob

    probs_outputs/=N_FOLDS
    return probs_outputs


probs = predict_regime(X_val)


CLASS_NAMES = ["bulk", "sb+", "sb-", "far+", "near", "far-"]   # must match your 0..5 label encoding

def bias_table(y, probs, raw_oof=None, b=None, class_names=CLASS_NAMES):
    y = np.asarray(y).ravel().astype(int)
    k = probs.shape[1]
    cols = {"class": class_names[:k]}
    if b is not None:
        cols["offset_b_k"] = np.round(np.asarray(b, float), 3)
    if raw_oof is not None:
        cols["pred_before_%"] = 100 * softmax(raw_oof, axis=1).mean(0)
    cols["pred_after_%"] = 100 * probs.mean(0)
    cols["observed_%"] = 100 * np.bincount(y, minlength=k) / len(y)
    cols["after/obs"] = cols["pred_after_%"] / cols["observed_%"]
    with pl.Config(tbl_rows=-1, float_precision=4):
        print(pl.DataFrame(cols))

bias_table(y_val, probs)


#=================================Fit Bulk Region============================#
# Want to be able to sample errors from cont distribution, outside of ones present 
# in training set. Then, we save the residuals for future sampling.

# The bulk error distribution is actually dependent on range, and follows a negative
# linear correlation

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats

LOW = -0.52
HIGH = 0.82
mid = (LOW + HIGH) / 2

def fit_bulk(features, labels, print_stats = False):
    df = pl.concat([features, labels], how = "horizontal")
    bulk = df.filter(
        pl.col("region") == "bulk"
    )
    X = bulk.select("bistatic_range_m")
    y = bulk["range_error_m"]

    #weird line that we don't want
    mask = ~((X[:, 0] < 110) & (np.abs(y) < 0.03))
    X = X.filter(mask)
    y = y.filter(mask)

    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    residuals = y - model.predict(X)
    residuals_test = y_test - model.predict(X_test)
    if print_stats:
        print(f"Slope (Coefficient): {model.coef_}")
        print(f"Intercept: {model.intercept_}")
        print(f"R² Score: {r2_score(y_test, residuals_test)}")


        q_lo, q_hi = np.quantile(residuals, [0.005, 0.995])
        width = (q_hi - q_lo) / 0.99
        lo = (q_lo + q_hi) / 2 - width / 2
        stat, p = stats.kstest(residuals, "uniform", args=(lo, width))
        
        q_lo, q_hi = np.quantile(residuals_test, [0.005, 0.995])
        width = (q_hi - q_lo) / 0.99
        lo = (q_lo + q_hi) / 2 - width / 2
        stat_robust, p_robust = stats.kstest(residuals_test, "uniform", args=(lo, width))
        print(f"KS Test D = {stat_robust:.4f}, p = {p_robust:.3g}")

    return model

def build_residual_distribution(model, X, y):
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.2, random_state = 42)
    preds = model.predict(X_train)
    residuals = y_train - preds 

    return residuals

def sample_bulk(model, bistatic_range, distribution):
    slope = model.coef_
    intercept = model.intercept_

    sampled_residual = random.choice(distribution)

    p = intercept + slope*bistatic_range + sampled_residual 
    return p



#======================================USAGE==============================#



