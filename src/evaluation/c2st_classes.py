from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import numpy as np
import polars as pl

#====================================Classifier Two Sample Test=======================#
# Train a discriminator to tell real rows (features, ISAC detected) from synthetic rows (same features, sampled detected)
# If the generator matches the real conditional distribution, the discriminator can't beat chance (AUC ≈ 0.5)
# Out of fold predictions via GroupKFold by scene, real/synthetic copies of a point always share a fold

C2ST_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,

    # small data, cpu is faster than cuda here (~0.8s vs ~34s per 5 fold fit)
    # cap threads, the default (all cores) oversubscribes on WSL and is ~60x slower
    'device': 'cpu',
    'num_threads': 6,
}


class C2ST:
    def __init__(self, params = C2ST_PARAMS, n_folds : int = 5, num_boost_round : int = 200):
        self.params = params
        self.n_folds = n_folds
        self.num_boost_round = num_boost_round
        self.models = []

    # real, synth : DataFrames with the same columns, groups : scene id per row (shared by real and synth)
    def fit(self, real : pl.DataFrame, synth : pl.DataFrame, groups):
        X = pl.concat([real, synth], how = "vertical")
        y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))])
        groups = np.concatenate([groups, groups])
        feature_names = X.columns
        X = X.to_numpy().astype(float)

        self.models = []
        self.oof = np.zeros(len(y))
        gain = np.zeros(len(feature_names))

        gkf = GroupKFold(n_splits = self.n_folds)
        for train_idx, test_idx in gkf.split(X, y, groups = groups):
            dtrain = lgb.Dataset(X[train_idx], label = y[train_idx], feature_name = feature_names)
            model = lgb.train(params = self.params, train_set = dtrain, num_boost_round = self.num_boost_round)

            self.oof[test_idx] = model.predict(X[test_idx])
            gain += model.feature_importance(importance_type = "gain")
            self.models.append(model)

        self.y = y
        self.auc = roc_auc_score(y, self.oof)
        self.acc = np.mean((self.oof >= 0.5) == y)
        self.importance = pl.DataFrame({"feature" : feature_names, "gain" : gain / self.n_folds}).sort("gain", descending = True)
        return self

    # P(real), averaged over the fold models
    def predict_proba(self, X):
        X = np.asarray(X, dtype = float)
        return np.mean([model.predict(X) for model in self.models], axis = 0)
