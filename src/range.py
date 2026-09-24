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


FEATURES = [f.value for f in PosFeature]
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

def group_split(X, y, tags, test_size=0.2, seed=42):
    groups = tags.select(pl.struct(["scenario_id", "drop_id"]).hash()).to_series().to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(gss.split(X, y, groups=groups))
    return X[train], X[test], y[train], y[test], tags[train], tags[test]


def train_regime(X, y, tags, params=GBM_PARAMS):
    X_new, X_val, y_new, y_val, tags_new, _ = group_split(X, y, tags)
    X_train, X_test, y_train, y_test, _, _ = group_split(X_new, y_new, tags_new)

    dtrain = lgb.Dataset(X_train, label=y_train.to_numpy(), feature_name=FEATURES)
    dval   = lgb.Dataset(X_val,   label=y_val.to_numpy(),   reference=dtrain, feature_name=FEATURES)

    model = lgb.train(
        params, dtrain,
        valid_sets=[dtrain, dval], valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(period=10), lgb.early_stopping(200)],
    )
    return model, (X_test, y_test)

model = train_regime(features, labels, tags)


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



