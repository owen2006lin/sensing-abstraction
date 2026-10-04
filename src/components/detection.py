from src.data.load import load_indep, load_pairs, group_split, tt_split, FEATURE_COLS, SHARED_FEATURES
from src.components.base import Component
from src.feature_building.feature_selector import indep_mutual_split, process_pair_features, process_pair_inference
import lightgbm as lgb
import numpy as np
import polars as pl
from src.data.features import IndepFeature, PairFeature

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

MATCH_COLS = [
    "scenario_id",
    "drop_id",
    "nn_norm_range_sep"
]

CALLBACKS = [
    # stops after 200 rounds of no improvement
    lgb.early_stopping(stopping_rounds = 200), 
    lgb.log_evaluation(period = 20)]

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
        samples = (rng.uniform(size = X.shape) < X).astype(int)
        df = pl.DataFrame(samples, schema = ["detected"])
        return df.with_columns(pl.col("detected").cast(pl.Boolean))

tags, labels, features = load_indep()
[X, X_val, y, y_val] = group_split(tags, labels, features)



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
        df = pl.DataFrame(sampled_indices, schema = ["score"])
        df = df.with_columns(
            pl.col("score").is_in([1,3]).alias("a_detected"),
            pl.col("score").is_in([2,3]).alias("b_detected")
        )
        return df.select(["a_detected", "b_detected"])



#======================================Full Classifier=======================#
from src.data.load import split_indep, split_pairs, load_class




class FullClassifier(Component):
    def __init__(self, models = None, params = [INDEP_PARAMS, PAIR_PARAMS], callbacks = [CALLBACKS,CALLBACKS]):
        if not models:
            self.indep_model = IndepClassifier()
            self.pair_model = PairClassifier()
        else:
            self.indep_model = models[0]
            self.pair_model = models[1]

        self.indep_callbacks = callbacks[0]
        self.pair_callbacks = callbacks[1]
        self.indep_params = params[0]
        self.pair_params = params[1]

    def fit(self, train_df, val_df, params = None, callbacks = None):
        if params is not None:
            self.indep_params = params[0]
            self.pair_params = params[1]
        if callbacks is not None:
            self.indep_callbacks = callbacks[0]
            self.pair_callbacks = callbacks[1]


        mutual_pairs, non_mutual = indep_mutual_split(train_df, MATCH_COLS)
        mutual_pairs_val, non_mutual_val = indep_mutual_split(val_df, MATCH_COLS)
        
        #indep, non mutual
        indep_features = [f.value for f in IndepFeature]
        tags, labels, features = split_indep(non_mutual, indep_features)
        tags_val, labels_val, features_val = split_indep(non_mutual_val, indep_features)

        if self.indep_model is None:
            indep_model = IndepClassifier(self.indep_params)
        else:
            indep_model = self.indep_model
        indep_model.fit(features, labels, features_val, labels_val, self.indep_params, self.indep_callbacks)
        self.indep_model = indep_model



        #pairs, mutual
        df = process_pair_features(mutual_pairs, MATCH_COLS)
        df_val = process_pair_features(mutual_pairs_val, MATCH_COLS)

        pair_features = FEATURE_COLS + SHARED_FEATURES
        tags, labels, features = split_pairs(df, pair_features)
        tags_val, labels_val, features_val = split_pairs(df_val, pair_features)

        if self.pair_model is None:
            pair_model = PairClassifier(self.pair_params)
        else:
            pair_model = self.pair_model
        pair_model.fit(features, labels, features_val, labels_val, self.pair_params, self.pair_callbacks)
        self.pair_model = pair_model
        return self

    def predict_proba(self, X):
        mutual_pairs, non_mutual = indep_mutual_split(X, match_cols= MATCH_COLS)

        features = non_mutual.select([f.value for f in IndepFeature])
        indep_preds = self.indep_model.predict_proba(features)

        pairs = process_pair_inference(mutual_pairs, MATCH_COLS) 
        pair_preds = self.pair_model.predict_proba(pairs)
        return indep_preds, pair_preds
            
        
    def sample(self, indep_preds, pair_preds, rng):
        indep_samples = self.indep_model.sample(indep_preds, rng)
        pair_samples = self.pair_model.sample(pair_preds, rng)
        return indep_samples, pair_samples

    def reassemble(self, X, indep_samples, pair_samples):
        merged_pair = pair_samples.select(
            pl.concat_list("a_detected", "b_detected").alias("detected")
        ).explode("detected")

        df_idx = X.with_row_index("_row")
        mutual, indep = indep_mutual_split(df_idx, MATCH_COLS)

        labels = pl.concat([
            mutual.select("_row").with_columns(detected=pl.Series(merged_pair)),
            indep.select("_row").with_columns(detected=pl.Series(indep_samples)),
        ])

        df_out = (
            df_idx.join(labels, on="_row", how="left")
                .sort("_row")   # joins don't guarantee order, so sort explicitly
                .drop("_row")
        )

        return df_out


