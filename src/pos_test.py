from __future__ import annotations
from src.load import *
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
 



 
tags, labels, features = load_pos()
range_error = labels["range_error_m"]
DEFAULT_SPLIT =  train_test_split(features, range_error, test_size=0.2, random_state=42)
[X_train, X_val, y_train, y_val] = DEFAULT_SPLIT
FEATURES = [f.value for f in PosFeature]


"""
Bulk-vs-tail diagnostics for a heavy-tailed regression target.
 
Three things, in order:
  1. sst_concentration      -- how much of SST the top q% of |y - ybar| owns
  2. report                 -- R2 / Spearman / calibration slope, full vs bulk vs tail
  3. bulk_refit_test        -- the load-bearing test: does training on the bulk
                              alone recover bulk signal the full-data model missed?
 
Drop-in for:
    tags, labels, features = load_pos()
    range_error = labels["range_error_m"]
    X_train, X_val, y_train, y_val = DEFAULT_SPLIT
    FEATURES = [f.value for f in PosFeature]
 
Polars in, numpy internally. See the RUN block at the bottom.
"""
 

 
# ----------------------------------------------------------------------
# coercion
# ----------------------------------------------------------------------
def _X(x, features=None) -> np.ndarray:
    """polars/pandas/numpy 2-D -> float ndarray. Assumes numeric features;
    if you have categoricals, hand LightGBM a pandas frame instead and
    swap the fit helper below."""
    if pl is not None and isinstance(x, pl.DataFrame):
        if features is not None:
            x = x.select(features)
        return x.to_numpy().astype(np.float64)
    if hasattr(x, "to_numpy"):
        if features is not None and hasattr(x, "columns"):
            x = x[features]
        return np.asarray(x.to_numpy(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)
 
 
def _y(x) -> np.ndarray:
    """anything 1-D -> float ndarray, raveled."""
    if pl is not None and isinstance(x, (pl.Series, pl.DataFrame)):
        x = x.to_numpy()
    elif hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.asarray(x, dtype=np.float64).ravel()
 
 
# ----------------------------------------------------------------------
# 1. where does SST actually live?
# ----------------------------------------------------------------------
def sst_concentration(y, qs=(0.001, 0.005, 0.01, 0.02, 0.05), label="y"):
    """Share of total sum of squares owned by the most extreme q fraction
    of points (by |y - ybar|). If the 1% row is > ~0.5, R2 is a tail metric
    and tells you almost nothing about the bulk."""
    y = _y(y)
    n = len(y)
    dev2 = (y - y.mean()) ** 2
    sst = dev2.sum()
    order = np.argsort(-dev2)
 
    print(f"\n=== SST concentration [{label}]  n={n}  var={y.var():.4g} ===")
    print(f"{'top q':>8} {'k pts':>7} {'share of SST':>13} {'|y-ybar| cutoff':>16}")
    out = {}
    for q in qs:
        k = max(1, int(round(q * n)))
        share = dev2[order[:k]].sum() / sst
        cut = np.sqrt(dev2[order[k - 1]])
        out[q] = share
        print(f"{q:>8.3%} {k:>7d} {share:>13.1%} {cut:>16.4g}")
    return out
 
 
# ----------------------------------------------------------------------
# 2. metrics split by bulk / tail
# ----------------------------------------------------------------------
def make_bulk_mask(y_ref, trim=0.01):
    """Define the bulk ONCE from a reference set (use train), then apply the
    same center+cutoff everywhere so the bulk doesn't drift between sets."""
    y_ref = _y(y_ref)
    center = float(y_ref.mean())
    cutoff = float(np.quantile(np.abs(y_ref - center), 1.0 - trim))
 
    def in_bulk(y):
        return np.abs(_y(y) - center) <= cutoff
 
    return center, cutoff, in_bulk
 
 
def _calib_slope(y, yhat):
    """OLS slope of y on yhat. ~1 = calibrated, ~0 = predictions carry no
    usable ordering, >1 = predictions under-dispersed relative to signal."""
    if len(y) < 3 or np.std(yhat) < 1e-12:
        return np.nan
    return float(np.polyfit(yhat, y, 1)[0])
 
 
def _block(y, yhat, name):
    n = len(y)
    if n < 3:
        print(f"  {name:<10} n={n:<7d} (too few points)")
        return {}
    r2 = r2_score(y, yhat)
    rho = spearmanr(y, yhat).statistic if np.std(yhat) > 1e-12 else np.nan
    slope = _calib_slope(y, yhat)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))
    print(
        f"  {name:<10} n={n:<7d} R2={r2:>8.4f}  rho={rho:>7.4f}  "
        f"slope={slope:>7.4f}  RMSE={rmse:>8.4g}  MAE={mae:>8.4g}"
    )
    return dict(n=n, r2=r2, spearman=rho, calib_slope=slope, rmse=rmse, mae=mae)
 
 
def report(y, yhat, in_bulk, label="val", deciles=True):
    """R2 / Spearman / calibration slope on all points, bulk only, tail only.
 
    R2 within the bulk is scored against the BULK's own mean, so it answers
    'do you beat a constant in here?' rather than borrowing credit from the tail.
    """
    y, yhat = _y(y), _y(yhat)
    m = in_bulk(y)
 
    print(f"\n=== metrics [{label}] ===")
    res = {
        "all": _block(y, yhat, "all"),
        "bulk": _block(y[m], yhat[m], "bulk"),
        "tail": _block(y[~m], yhat[~m], "tail"),
    }
    if res["tail"].get("n", 0) < 30:
        print(f"  ! only {res['tail'].get('n', 0)} tail points -- tail numbers are noise")
 
    if deciles and m.sum() > 100:
        yb, pb = y[m], yhat[m]
        edges = np.quantile(pb, np.linspace(0, 1, 11))
        edges[-1] += 1e-9
        idx = np.clip(np.digitize(pb, edges[1:-1]), 0, 9)
        print(f"\n  bulk decile lift (flat mean(y) column == no bulk signal)")
        print(f"  {'bin':>4} {'n':>6} {'mean pred':>11} {'mean y':>11} {'sd y':>10}")
        for b in range(10):
            s = idx == b
            if s.sum():
                print(
                    f"  {b:>4d} {s.sum():>6d} {pb[s].mean():>11.4g} "
                    f"{yb[s].mean():>11.4g} {yb[s].std():>10.4g}"
                )
        spread = yb[idx == 9].mean() - yb[idx == 0].mean()
        print(f"  top-bottom decile spread in y: {spread:.4g}  (sd(y_bulk)={yb.std():.4g})")
 
    return res
 
 
# ----------------------------------------------------------------------
# 3. the load-bearing test: refit on the bulk only
# ----------------------------------------------------------------------
def _default_fit(Xtr, ytr, objective="l2", seed=0, **kw):
    """Swap this for your real model config -- the comparison is only
    meaningful if both arms use the settings you actually ship."""
    params = dict(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(kw)
    try:
        from lightgbm import LGBMRegressor
 
        obj = {"l2": "regression", "huber": "huber", "l1": "regression_l1"}[objective]
        return LGBMRegressor(objective=obj, **params).fit(Xtr, ytr)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
 
        loss = {"l2": "squared_error", "huber": "absolute_error", "l1": "absolute_error"}[objective]
        return HistGradientBoostingRegressor(
            loss=loss,
            learning_rate=params["learning_rate"],
            max_iter=params["n_estimators"],
            max_leaf_nodes=params["num_leaves"],
            min_samples_leaf=params["min_child_samples"],
            random_state=seed,
        ).fit(Xtr, ytr)
 
 
def bulk_refit_test(
    X_train, y_train, X_val, y_val,
    features=None, trim=0.01, objective="l2", fit_fn=_default_fit, seed=0,
):
    """Train on everything vs train on the bulk only; score BOTH on the same
    held-out bulk.
 
      bulk-only model clearly better on val-bulk  -> the tail is hijacking the
          loss. Split the problem (separate tail head, or robust/quantile loss
          for the bulk).
      both ~0 on val-bulk                          -> the bulk is genuinely
          unpredictable from these features. Nothing to fix upstream; spend the
          effort on the spread/shape model instead.
    """
    Xtr, Xva = _X(X_train, features), _X(X_val, features)
    ytr, yva = _y(y_train), _y(y_val)
 
    center, cutoff, in_bulk = make_bulk_mask(ytr, trim=trim)
    mtr, mva = in_bulk(ytr), in_bulk(yva)
    print(
        f"\n=== bulk refit test  trim={trim:.2%}  objective={objective} ===\n"
        f"  bulk = |y - {center:.4g}| <= {cutoff:.4g}\n"
        f"  train {mtr.sum()}/{len(ytr)} in bulk   val {mva.sum()}/{len(yva)} in bulk"
    )
 
    m_full = fit_fn(Xtr, ytr, objective=objective, seed=seed)
    m_bulk = fit_fn(Xtr[mtr], ytr[mtr], objective=objective, seed=seed)
 
    p_full, p_bulk = m_full.predict(Xva), m_bulk.predict(Xva)
 
    r_full = report(yva, p_full, in_bulk, label="full-data model", deciles=False)
    r_bulk = report(yva, p_bulk, in_bulk, label="bulk-only model", deciles=True)
 
    a = r_full.get("bulk", {}).get("r2", np.nan)
    b = r_bulk.get("bulk", {}).get("r2", np.nan)
    print(f"\n  >> val-bulk R2:  full-data {a:.4f}   bulk-only {b:.4f}   delta {b - a:+.4f}")
    if np.isfinite(b) and b > max(0.02, a + 0.02):
        print("  >> bulk signal exists and the tail was eating it. Split the problem.")
    elif np.isfinite(b) and b < 0.02:
        print("  >> no bulk signal either way. Step 1 is doing all it can; fix step 3.")
    else:
        print("  >> inconclusive/marginal -- rerun on grouped OOF preds, not one split.")
 
    return dict(full=r_full, bulk=r_bulk, models=(m_full, m_bulk), mask=(mtr, mva))
 
 
# ======================================================================
# RUN  -- edit the two marked lines
# ======================================================================
if __name__ == "__main__":
    # ---- 1. is R2 a tail metric here? -------------------------------
    sst_concentration(y_train, label="y_train")
    sst_concentration(y_val, label="y_val")
 
    # ---- 2. your CURRENT model, bulk vs tail ------------------------
    # >>> EDIT: your existing fitted GBM
    model = lgb.Booster(model_file = "models/position_bias.txt")

    yhat_val = model.predict(_X(X_val, FEATURES))
    _, _, in_bulk = make_bulk_mask(y_train, trim=0.01)
    report(y_val, yhat_val, in_bulk, label="current model / val")
 
    # ---- 3. the load-bearing test -----------------------------------
    for trim in (0.005, 0.01, 0.02):
        bulk_refit_test(
            X_train, y_train, X_val, y_val,
            features=FEATURES,   # >>> EDIT: drop if X_* are already numeric arrays
            trim=trim,
            objective="l2",      # then rerun with "huber"
        )