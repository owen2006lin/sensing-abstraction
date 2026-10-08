from src.components.base import Component
from src.data.features import PosFeature
import lightgbm as lgb
import numpy as np
import polars as pl

# 3D position error label from load_pos()
ERROR_COL = "position_error_3d_m"

#====================================B2 : Direct Quantile GBM=======================#
# Predicts the 3D position error straight from the features, no range / angle decomposition or geometry
# One LightGBM quantile regressor per level in TAUS, fit on log(error) : quantiles are invariant under monotone
# transforms, so exp() of the log quantiles are the error quantiles, and samples stay positive
# Sampling : inverse CDF, u ~ U(0, 1) linearly interpolated between the predicted quantiles
# u outside [TAUS[0], TAUS[-1]] is clamped to the outer quantiles, so the extreme 1% of tails is truncated

TAUS = np.array([0.005, 0.01, 0.025, *np.round(np.arange(0.05, 0.951, 0.05), 2), 0.975, 0.99, 0.995])

QUANTILE_PARAMS = {
    "objective": "quantile",       # alpha set per level in fit
    "metric": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,

    # small data, cpu, capped threads (same reasoning as C2ST_PARAMS)
    'device': 'cpu',
    'num_threads': 6,
}


class QuantileErrorModel(Component):
    def __init__(self, params = QUANTILE_PARAMS, taus = TAUS, num_boost_round : int = 1000, early_stopping : int = 50):
        self.params = params
        self.taus = np.asarray(taus)
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.models = []
        self.feature_names = [f.value for f in PosFeature]

    # X : features DataFrame (needs PosFeature columns), y : ERROR_COL
    # X_val / y_val : optional held out set for early stopping, otherwise num_boost_round for every level
    def fit(self, X, y, X_val = None, y_val = None, params = None, callbacks = None):
        if params is not None:
            self.params = params

        dtrain = lgb.Dataset(X.select(self.feature_names).to_numpy().astype(float),
                             label = np.log(y[ERROR_COL].to_numpy()), feature_name = self.feature_names,
                             free_raw_data = False)
        valid_sets = None
        if X_val is not None and y_val is not None:
            valid_sets = [lgb.Dataset(X_val.select(self.feature_names).to_numpy().astype(float),
                                      label = np.log(y_val[ERROR_COL].to_numpy()), reference = dtrain)]
            callbacks = [lgb.early_stopping(self.early_stopping, verbose = False)] + (callbacks or [])

        self.models = []
        for tau in self.taus:
            model = lgb.train(params = {**self.params, "alpha": tau}, train_set = dtrain,
                              num_boost_round = self.num_boost_round, valid_sets = valid_sets, callbacks = callbacks)
            self.models.append(model)
        return self

    # (n_points, len(taus)) error quantiles. Sorted per row so crossed quantiles are rearranged into a valid CDF
    def predict_proba(self, X):
        X = X.select(self.feature_names).to_numpy().astype(float)
        log_q = np.column_stack([model.predict(X, num_iteration = model.best_iteration) for model in self.models])
        return np.exp(np.sort(log_q, axis = 1))

    # X : quantiles from predict_proba, interpolated in log space so it matches the fitted (log error) quantiles
    def sample(self, X, rng):
        log_q = np.log(np.asarray(X))
        u = np.clip(rng.uniform(size = len(log_q)), self.taus[0], self.taus[-1])

        hi = np.clip(np.searchsorted(self.taus, u), 1, len(self.taus) - 1)
        lo = hi - 1
        w = (u - self.taus[lo]) / (self.taus[hi] - self.taus[lo])

        rows = np.arange(len(log_q))
        err = np.exp((1 - w) * log_q[rows, lo] + w * log_q[rows, hi])
        return pl.DataFrame({"err_3d" : err})
