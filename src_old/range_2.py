from src.load import *
import numpy as np








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

FEATURES = [f.value for f in PosFeature]
GROUP_COLS = ["scenario_id", "drop_id"]
N_FOLDS = 5
SEED = 0



#=================================Training Sequence=============================#

from scipy.optimize import minimize
from scipy.special import softmax
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats


REGION_NAMES = ["bulk", "sideband+", "sideband-", "far+", "far-", "near"]   # index = label 0..5
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


def predict_out_of_fold(features : pl.DataFrame, labels : pl.DataFrame, tags : pl.DataFrame, params: dict):
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



def fit_bulk(bulk, print_stats = False):

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

def load_class_model(path):
    dir_path = Path(path)
    oof_logits = np.genfromtxt(dir_path / "oof_logits.csv" )
    oof_cal = np.genfromtxt(dir_path / "oof_cal.csv" )
    b_final = np.genfromtxt(dir_path / "b_final.csv" )
    models = []

    for i in range(N_FOLDS):
        model = lgb.Booster(model_file= dir_path / f"models/regime_class_{i}.txt")
        models.append(model)
    return oof_logits, oof_cal, b_final, models
    
#===================================Inference Sequence==========================#
import random

def predict_regime(features, bias, models):
    for model in models:
        preds = model.predict(features, raw_score = True)
        prob = apply_bias(preds, bias)
    prob/=N_FOLDS
    return prob

CLASS_NAMES = ["bulk", "sb+", "sb-", "far+", "far-", "near"]  

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

def sample_bulk_single(model, bistatic_range, distribution):
    slope = model.coef_
    intercept = model.intercept_

    sampled_residual = random.choice(distribution)

    p = intercept + slope*bistatic_range + sampled_residual 
    return p

def sample_bulk(bulk_inputs, bulk_model, bulk_distribution):
    bulk_outputs = bulk_inputs.with_columns(
        pl.col("bistatic_range_m")
        .map_elements(lambda r: float(np.ravel(sample_bulk_single(bulk_model, r, bulk_distribution))[0]),
                        return_dtype=pl.Float64)
        .alias("sampled_error")
    ).select("sampled_error")
    return bulk_outputs


def build_distributions(features, labels, regions):
    df = pl.concat([features, labels], how = "horizontal")
    distributions = []
    for region in regions:
        distr = df.filter(
            pl.col("region_txt") == region
        )
        if region == "far+":
            distr = distr.with_columns(
                pl.col("nn_norm_range_sep")
                .cast(pl.Float64)
                .fill_null(float("inf"))
                .qcut(5, labels=["1", "2", "3", "4", "5"], allow_duplicates=True)
                .cast(pl.String)
                .cast(pl.Int8)
                .alias("quintile"))
        distributions.append(distr)
    return distributions


def jitter(error, type):
    rng = np.random.default_rng()
    match type:
        #sideband+
        case 1:
            bounds = [1.4, 1.85]
            jitter = rng.uniform(low = -0.03, high = 0.03)
            val = error + jitter
            if val >= bounds[0] and val <= bounds[1]:
                return val
            else: return error - jitter
        #sideband-
        case 2:
            bounds = [-2.25, -1.55]
            jitter = rng.uniform(low = -0.06, high = 0.06)
            val = error + jitter
            if val >= bounds[0] and val <= bounds[1]:
                return val
            else: return error - jitter
        #far+
        case 3:
            bounds = [1.85]
            jitter = rng.uniform(low = -0.1, high = 0.1)
            val = error + jitter
            if val >= bounds[0]:
                return val
            else: return error - jitter
        #far-
        case 4:
            bounds = [-2.25]
            jitter = rng.uniform(low = -0.15, high = 0.15)
            val = error + jitter
            if val <= bounds[0]:
                return val
            else: return error - jitter
        #near
        case 5:
            upper = [0.82, 1.4]
            lower = [-1.55, -0.52]
            jitter = rng.uniform(low = -0.03, high = 0.03)
            val = error + jitter

            if error >= upper[0]:
                if val <= 1.4:
                    return val
                else: return error - jitter
            else:
                if val >= -1.55:
                    return val
                else: return error - jitter


def sample_non_bulk(data, distributions):
    ret = []
    for type, nn_dist in data:
        distr = distributions[type - 1]
        # Far+
        if type == 3:
            breaks = [distr.get_column("nn_norm_range_sep").quantile(p) for p in (0.2, 0.4, 0.6, 0.8)]
            q = 1 + sum(nn_dist > b for b in breaks)
            quintile_df = distr.filter(pl.col("quintile") == q)

            error = random.choice(quintile_df["range_error_m"].to_numpy().ravel())
        else:
            error = random.choice(distr["range_error_m"].to_numpy().ravel())
        error = jitter(error, type)
        ret.append(error)
    return ret



def sample_class(probs):
    preds = []
    for p in probs:
        #[p_bulk, p_sb_plus, p_sb_minus, p_far_plus, p_far_minus, p_near] = p
        idx = random.choices(range(len(p)), weights = p, k=1)[0]
        preds.append(idx)
    return  pl.Series("regime",preds)
