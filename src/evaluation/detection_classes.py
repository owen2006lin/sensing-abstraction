from src.components.base import Component
from src.data.features import IndepFeature
import numpy as np
import polars as pl
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Measurement space errors the baselines sample for detected points (columns from load_pos())
ERROR_COLS = ["range_error_m", "rx_azimuth_error_deg", "rx_elevation_error_deg"]

#====================================B1 : Constant Rate + Gaussian Error=======================#
# One detection rate for every point, errors drawn from a single multivariate gaussian fit on detected points
# y : DataFrame with "detected" and optionally error columns (null where missed)

class ConstantClassifier(Component):
    def __init__(self, rate = None, mean = None, cov = None, error_cols = None):
        self.rate = rate
        self.mean = mean
        self.cov = cov
        self.error_cols = error_cols if error_cols is not None else []

    def fit(self, X, y, X_val = None, y_val = None, params = None, callbacks = None):
        self.rate = y["detected"].mean()

        self.error_cols = [c for c in y.columns if c != "detected"]
        if self.error_cols:
            errors = y.filter(pl.col("detected") == 1).select(self.error_cols).drop_nulls().to_numpy()
            self.mean = errors.mean(axis = 0)
            self.cov = np.cov(errors, rowvar = False)
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.rate)

    def sample(self, X, rng, features = None):
        X = np.asarray(X)
        detected = rng.uniform(size = X.shape) < X
        df = pl.DataFrame({"detected" : detected})

        if self.error_cols:
            errors = rng.multivariate_normal(self.mean, self.cov, size = len(X))
            errors[~detected] = np.nan
            df = df.with_columns(pl.DataFrame(errors, schema = self.error_cols).fill_nan(None))
        return df



#====================================B4 : MLP Classifier=======================#
# sklearn MLP on IndepFeature columns with a sigmoid head for P(detected), no error model
# y : DataFrame with "detected" (any error columns are ignored)

MDN_PARAMS = {
    "hidden_layer_sizes" : (64, 64),
    "activation" : "tanh",
    "learning_rate_init" : 1e-3,
    "batch_size" : 512,
    "max_iter" : 200,
    "early_stopping" : True,      # holds out validation_fraction of the train rows (random, not grouped)
    "validation_fraction" : 0.1,
    "n_iter_no_change" : 20,
    "random_state" : 42,
}


class MDNClassifier(Component):
    def __init__(self, params = MDN_PARAMS):
        self.params = params
        self.model = None
        self.feature_names = [f.value for f in IndepFeature]

    # X : features DataFrame (needs IndepFeature columns), y : "detected"
    # X_val / y_val unused, sklearn early stopping uses its own split of X
    def fit(self, X, y, X_val = None, y_val = None, params = None, callbacks = None):
        if params is not None:
            self.params = params

        # standardize, mean impute + missing indicator for any column with nulls (eg nn_norm_range_sep with no neighbor)
        self.model = make_pipeline(
            SimpleImputer(add_indicator = True),
            StandardScaler(),
            MLPClassifier(**self.params)
        )
        self.model.fit(X.select(self.feature_names).to_numpy().astype(float), y["detected"].to_numpy().astype(int))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X.select(self.feature_names).to_numpy().astype(float))[:, 1]

    # X : probs from predict_proba
    def sample(self, X, rng, features = None):
        X = np.asarray(X)
        detected = rng.uniform(size = X.shape) < X
        return pl.DataFrame({"detected" : detected})



#====================================B5 : Split MDN (indep + mutual pairs)=======================#
# Same indep / mutual split as FullClassifier : MDNClassifier on non mutual points, and this pair net on mutual pairs
# sklearn MLP with a 4 way softmax over the joint (0 neither, 1 only b, 2 only a, 3 both)
# Use as FullClassifier(models = [MDNClassifier(), PairMDNClassifier()], params = [MDN_PARAMS, PAIR_MDN_PARAMS], callbacks = [None, None])

PAIR_MDN_PARAMS = {
    "hidden_layer_sizes" : (64, 64),
    "activation" : "tanh",
    "learning_rate_init" : 1e-3,
    "batch_size" : 512,
    "max_iter" : 200,
    "early_stopping" : True,      # holds out validation_fraction of the train rows (random, not grouped)
    "validation_fraction" : 0.1,
    "n_iter_no_change" : 20,
    "random_state" : 42,
}


class PairMDNClassifier(Component):
    def __init__(self, params = PAIR_MDN_PARAMS):
        self.params = params
        self.model = None

    # X : pair features DataFrame (FEATURE_COLS + SHARED_FEATURES), y : "score" = 2a + b
    # X_val / y_val unused, sklearn early stopping uses its own split of X
    def fit(self, X, y, X_val = None, y_val = None, params = None, callbacks = None):
        if params is not None:
            self.params = params

        # standardize, mean impute + missing indicator for any column with nulls
        self.model = make_pipeline(
            SimpleImputer(add_indicator = True),
            StandardScaler(),
            MLPClassifier(**self.params)
        )
        self.model.fit(X.to_numpy().astype(float), y["score"].to_numpy().astype(int))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X.to_numpy().astype(float))

    # Same output format as PairClassifier.sample, so FullClassifier.reassemble works unchanged
    def sample(self, X, rng):
        X = np.asarray(X)
        score = (X.cumsum(axis = 1) > rng.uniform(size = (len(X), 1))).argmax(axis = 1)
        return pl.DataFrame({
            "a_detected" : np.isin(score, [2, 3]),
            "b_detected" : np.isin(score, [1, 3])
        })
