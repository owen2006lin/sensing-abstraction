from src.data.load import load_indep, load_pairs, group_split, tt_split
from src.components.base import Component
import lightgbm as lgb
import numpy as np

INDEP_PARAMS = {
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

PAIR_PARAMS = {
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


#====================================Independent Classifier=======================#

class IndepClassifier(Component):
    def __init__(self, model = None, params = INDEP_PARAMS):
        self.params = params
        self.model = model

    def fit(self, X, y, X_val = None , y_val = None, params = None, callbacks = None):
        if params is not None:
            self.params = params
        
        feature_names = X.columns
        dtrain = lgb.Dataset(X, label = y.to_numpy(), feature_name = feature_names)
        valid_sets = [dtrain]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label = y_val.to_numpy(), feature_name = feature_names, reference = dtrain)
            valid_sets.append(dval)
            valid_names.append('valid')

        self.model = lgb.train(
            params = self.params,
            train_set = dtrain,
            valid_sets = valid_sets,
            valid_names = valid_names,
            callbacks=callbacks
        )
        return self
    
    def predict_proba(self,X):
        preds = self.model.predict(X)
        return preds

    def sample(self, X, rng):
        X = np.asarray(X)
        return (rng.uniform(size = X.shape) < X).astype(int)

tags, labels, features = load_indep()
[X, X_val, y, y_val] = group_split(tags, labels, features)

#classifier = IndepClassifier()
#classifier.fit(X, y)
#classifier.save("models/indep_classifier.joblib")

classifier = IndepClassifier.load("models/indep_classifier.joblib")
preds = classifier.predict_proba(X_val)
print(preds)
rng = np.random.default_rng(seed = 42)
print(classifier.sample(preds, rng))

#====================================Pair Classifier=======================#
class PairClassifier(Component):
    def __init__(self, model = None, params = PAIR_PARAMS):
            self.params = params
            self.model = model
    
    def fit(self, X, y, X_val = None , y_val = None, params = None, callbacks = None):
        if params is not None:
            self.params = params

        feature_names = X.columns
        dtrain = lgb.Dataset(X, label = y.to_numpy(), feature_name = feature_names)
        valid_sets = [dtrain]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label = y_val.to_numpy(), feature_name = feature_names, reference = dtrain)
            valid_sets.append(dval)
            valid_names.append('valid')

        self.model = lgb.train(
            params = self.params,
            train_set = dtrain,
            valid_sets = valid_sets,
            valid_names = valid_names,
            callbacks=callbacks
        )
        return self
    
    def predict_proba(self,X):
        preds = self.model.predict(X)
        return preds

    def sample(self, X, rng):
        X = np.asarray(X)
        sampled_indices = [rng.choice(len(row), p = row) for row in X]
        return sampled_indices

CALLBACKS = [
    # stops after 200 rounds of no improvement
    lgb.early_stopping(stopping_rounds = 200), 
    lgb.log_evaluation(period = 20)]

tags, labels, features = load_pairs()
[X, X_val, y, y_val] = group_split(tags, labels, features)
#pair_classifier = PairClassifier()
#pair_classifier.fit(X,y, callbacks=CALLBACKS)
#pair_classifier.save("models/pair_classifier.joblib")
pair_classifier = PairClassifier.load("models/pair_classifier.joblib")
preds = pair_classifier.predict_proba(X_val)
samples = pair_classifier.sample(preds, rng)

print(X_val)