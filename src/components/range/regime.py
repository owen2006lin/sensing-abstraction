from src.components.base import Component
from src.data.load import load_pos, group_split
from src.data.features import PosFeature
import polars as pl
import lightgbm as lgb
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize


GROUP_COLS = ["scenario_id", "drop_id"]
SEED = 0
REGION_NAMES = ["bulk", "sideband+", "sideband-", "far+", "far-", "near"]   # index = label 0..5
FEATURES = [f.value for f in PosFeature]
N_CLASSES = 6

GBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 6,                 # bulk, sb+, sb-, far+, far-, near  (encode labels as 0..5)
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


def label_regime(errors):
    e = pl.col("range_error_m")
    labels = (
        errors.select("range_error_m")
        .with_columns(
            pl.when(e.is_between(-0.52, 0.82)).then(0)
            .when(e.is_between(1.40, 1.85)).then(1)
            .when(e.is_between(-2.25, -1.55)).then(2)
            .when(e > 1.85).then(3)
            .when(e < -2.25).then(4)
            .otherwise(5)
            .cast(pl.Int32)
            .alias("region")
        )
        .with_columns(pl.col("region").replace_strict(list(range(6)), REGION_NAMES).alias("region_txt"))
    )
    return labels

def assign_folds(df : pl.DataFrame, n_folds : int, seed : int = SEED):
    folds = (
        df.select(GROUP_COLS)
        .unique()
        .sort(GROUP_COLS)
        .sample(fraction = 1.0, shuffle = True, seed = seed)
        .with_columns((pl.int_range(pl.len()) % n_folds).alias("fold"))
    )
    return df.join(folds, on = GROUP_COLS, how = "left", maintain_order = "left")


def fit_bias(raw, y):                      # raw: (n, 6) held-out raw scores, y: labels 0..5
    def nll(b):
        q = softmax(raw + np.r_[0.0, b], axis=1)
        return -np.log(q[np.arange(len(y)), y] + 1e-12).mean()
    return np.r_[0.0, minimize(nll, np.zeros(raw.shape[1] - 1), method="L-BFGS-B").x]

def apply_bias(raw, b):
    return softmax(raw + b, axis=1)



class RegimeClassifier(Component):
    def __init__(self, models = None, bias = None, n_models = 5):
        if not models:
            models = []
            for i in range(n_models):
                models.append(lgb.LGBMClassifier())
            self.models = models
        else:
            self.models = models
        self.bias = bias


    def fit(self, train_df, val_df = None, params = GBM_PARAMS):
        n_models = len(self.models)
        tags = assign_folds(train_df.select(GROUP_COLS), n_models)

        X = train_df.select(FEATURES).to_numpy()
        y = label_regime(train_df.select("range_error_m"))["region"].to_numpy()
        fold = tags["fold"].to_numpy()

        if val_df is not None:
            X_val = val_df.select(FEATURES).to_numpy()
            y_val = label_regime(val_df.select("range_error_m"))["region"].to_numpy()

        models = []
        out_of_fold = np.full((X.shape[0], N_CLASSES), np.nan)

        for k in range(n_models): 
            print(f"Cross fitting fold {k}")
            (train, valid) = (fold!= k, fold == k)  
            dtrain = lgb.Dataset(X[train], label = y[train], feature_name = FEATURES)
            
            if val_df is None:
                print("Warning, no val_df given. Early stopping determined by train_df")
                dvalid = dtrain
            else:
                dvalid = lgb.Dataset(X_val, label = y_val, feature_name = FEATURES, reference = dtrain)
            
            model = lgb.train(params, dtrain, 
                            valid_sets = [dvalid],
                            valid_names = ["valid"],
                            callbacks=[lgb.log_evaluation(period=50),
                                        lgb.early_stopping(stopping_rounds=100, verbose=True)])
            print(model.best_iteration)
            out_of_fold[valid] = model.predict(X[valid], raw_score = True)
            models.append(model)
        self.models = models
        #==============================Cross fit bias, fit on 4 folds then apply to 5th=======================#
        y_int = y.ravel().astype(int)
        oof_cal = np.full_like(out_of_fold, np.nan)
        biases = []
        for k in range(n_models):
            (train, valid) = (fold != k, fold == k)
            b_k = fit_bias(out_of_fold[train], y_int[train])
            oof_cal[valid] = apply_bias(out_of_fold[valid], b_k)
            biases.append(b_k)
            print(f"fold {k} bias: {np.round(b_k, 3)}")

        b_final = fit_bias(out_of_fold, y_int)    #fit on all OOF rows
        self.bias = b_final
        return out_of_fold, oof_cal, b_final, models

    def predict_proba(self, X):
        bias = self.bias
        probs = 0
        for model in self.models:
            preds = model.predict(X, raw_score = True)
            probs += apply_bias(preds, bias)
        probs/=len(self.models)
        return probs
    
    def sample(self, probs, rng):
        preds = [rng.choice(len(p), p=p) for p in probs]
        df = pl.DataFrame({"region" : preds})
        df = df.with_columns(pl.col("region").replace_strict(list(range(6)), REGION_NAMES).alias("region_txt"))
        return df


def save_class_outputs(oof_logits, oof_cal, b_final, models, path):
    dir_path = Path(path)
    dir_path.mkdir(parents = True, exist_ok = True)

    np.savetxt(dir_path / "oof_logits.csv", oof_logits)
    np.savetxt(dir_path / "oof_cal.csv", oof_cal)
    np.savetxt(dir_path / "b_final.csv", b_final)

    model_path = Path(dir_path / "models")
    model_path.mkdir(parents = True, exist_ok = True)

    i = 0
    for model in models:
        model.save_model(dir_path / f"models/regime_class_{i}.txt")
        i+=1

#save_class_outputs(oof_logits, oof_cal, b_final, models, "test_range_1")

from pathlib import Path

def load_class_model(path):
    dir_path = Path(path)
    oof_logits = np.genfromtxt(dir_path / "oof_logits.csv" )
    oof_cal = np.genfromtxt(dir_path / "oof_cal.csv" )
    b_final = np.genfromtxt(dir_path / "b_final.csv" )
    models = []

    for i in range(5):
        model = lgb.Booster(model_file= dir_path / f"models/regime_class_{i}.txt")
        models.append(model)
    return oof_logits, oof_cal, b_final, models

