"""
Conditional gaussian range-error model:  y | x ~ N(mu(x), sigma(x)^2).

Two LightGBM models -- one for the conditional mean, one for the conditional
log-variance -- plus one scalar bias correction. Everything here exists to make
sigma honest; the mean model is the easy half.

    from gaussian_error_model import GaussianErrorModel

    m = GaussianErrorModel().fit(X_train, y_train)
    mu, sigma = m.predict_mu_sigma(X_val)
    S = m.sample(X_val, n_draws=200)        # simulate range errors
    pit = m.cdf(y_val, X_val)               # calibration check

Diagnostics live in gaussian_error_tests.py; this file has no plotting and no
metrics so it can be imported into production without dragging matplotlib in.

SAMPLERS
--------
sampler="gaussian"   draws  mu + sigma * z,  z ~ N(0,1)
sampler="empirical"  draws  mu + sigma * z*, z* resampled from the normalized
                     residuals (y - mu)/sigma of the held-out slice

The empirical sampler keeps mu(x) and sigma(x) exactly as fitted and replaces
only the SHAPE. On this data the clean-row bulk is close to uniform (robust
sd(z) ~1.32 against 1.35 predicted for a uniform bulk), and the gaussian bell
leaks mass past the box edge -- which is why it over-rejects by ~42% at a
0.9 m gate. The resampled shape carries the box and the real blunder tail
with no distributional assumption. It is also invariant to k: k scales
sigma up and z* down by the same factor, so the draws do not depend on it.

n_strata > 1 builds a separate pool per band of predicted sigma. Use it if
sd(z) varies across sigma deciles (it rises to 1.85 / 2.6 in the top two
deciles on this data), so wide rows get a wide-row tail shape.

KNOWN AND ACCEPTED LIMITATIONS
------------------------------
The gaussian form is symmetric and thin-tailed; real range error is neither.
Measured against a 17-level quantile model on the same split, the costs are:

  * under-rejects by 20-28% at gate thresholds >= 1.5 m
  * injects ~3.7x the true bias among accepted errors at a 0.5 m gate
    (0.091 vs 0.024 m) -- this shrinks fast as the gate widens
  * cannot produce values below about -9 m when -13 m has been observed
  * PIT KS 0.078 vs 0.022 (both reject uniformity at n=9k)

In exchange it is 7 fits instead of 13+ and about a fifth of the code. At a
tight gate (tau ~ 0.97) these costs cost ~1% on end-to-end position error,
because the gate discards the region where the gaussian is wrong. That trade
stops holding if the gate is widened much past ~4 m -- re-run the tests if it
moves.
"""

from __future__ import annotations

import warnings

import numpy as np

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


# ======================================================================
# coercion
# ======================================================================
def _X(x, features=None) -> np.ndarray:
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
    if pl is not None and isinstance(x, (pl.Series, pl.DataFrame)):
        x = x.to_numpy()
    elif hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return np.asarray(x, dtype=np.float64).ravel()


def _holdout(n: int, frac: float, groups=None, rng=None) -> np.ndarray:
    """Boolean mask of length n. Holds out whole groups when groups is given,
    so correlated rows never straddle the split."""
    rng = rng or np.random.default_rng(0)
    m = np.zeros(n, bool)
    if frac <= 0:
        return m
    if groups is not None:
        g = np.asarray(groups)
        uq = np.unique(g)
        hold = set(rng.choice(uq, max(1, int(round(frac * len(uq)))), replace=False))
        return np.fromiter((gi in hold for gi in g), bool, n)
    m[rng.choice(n, max(1, int(round(frac * n))), replace=False)] = True
    return m


BASE_PARAMS = dict(
    n_estimators=3000,   # a ceiling, not a target -- early stopping picks the count
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
)

GPU_PARAMS = dict(device="cuda", gpu_use_dp=False)

# E[log chi^2_1]. Regressing log(r^2) under L2 estimates E[log r^2], which for
# gaussian residuals is log(sigma^2) - 1.2704, so exp() lands low by
# exp(-1.2704/2) = 0.53x. See `k_` below.
_LOG_CHI2_BIAS = 1.2704


class RecalibrationError(RuntimeError):
    """Raised when the sigma bias correction could not be fitted.

    Deliberately fatal. Without `k_` the model is uniformly ~47% overconfident,
    and nothing about the output looks wrong -- sigma is simply half what it
    should be on every row. That surfaces later as coverage failures that look
    like a modelling problem rather than a missing constant.
    """


class GaussianErrorModel:
    """mu(x) from an L2 model; sigma(x) from a log-variance model on
    out-of-fold residuals; one scalar to undo the log-chi^2 bias."""

    def __init__(self, n_splits=5, use_gpu=True, seed=0, params=None,
                 es_frac=0.15, require_recal=True, support=None,
                 support_margin=0.0, verbose=True, sampler="gaussian",
                 n_strata=1):
        """
        n_splits      folds used to produce out-of-fold residuals (see `fit`)
        es_frac       fraction held out for early stopping AND for fitting k
        require_recal raise if k could not be fitted, rather than shipping a
                      silently overconfident model
        support       (lo, hi) hard clamp on draws, in metres. None learns it
                      from the training labels. Pass a physical bound if the
                      radio gives you one better than the sample does.
        sampler       "gaussian" or "empirical" -- see module docstring. Both
                      are always fitted; this only sets the default used by
                      sample / cdf / predict_quantiles.
        n_strata      number of sigma bands for the empirical pool (1 = pooled)
        """
        if sampler not in ("gaussian", "empirical"):
            raise ValueError(f"sampler must be 'gaussian' or 'empirical', got {sampler!r}")
        self.sampler = sampler
        self.n_strata = int(n_strata)
        self.z_pools_ = None
        self.z_pgrids_ = None
        self.z_edges_ = None
        self.n_splits = n_splits
        self.use_gpu = use_gpu
        self.seed = seed
        self.params = dict(BASE_PARAMS if params is None else params)
        self.es_frac = es_frac
        self.require_recal = require_recal
        self.support = support
        self.support_margin = support_margin
        self.verbose = verbose
        self.k_ = None
        self.support_ = support
        self.features_ = None
        self.resid_sd_ = None
        self.r2_ = None

    # -- internals ----------------------------------------------------
    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _reg(self, Xtr, ytr, eval_set=None):
        """One LightGBM. Tries CUDA, falls back to CPU on any failure."""
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        p = dict(self.params, random_state=self.seed)
        kw = {}
        if eval_set is not None:
            kw = dict(eval_set=[eval_set],
                      callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.use_gpu:
                try:
                    return LGBMRegressor(objective="regression", **p, **GPU_PARAMS) \
                        .fit(Xtr, ytr, **kw)
                except Exception:  # noqa: BLE001
                    pass
            return LGBMRegressor(objective="regression", **p).fit(Xtr, ytr, **kw)

    # -- fit ----------------------------------------------------------
    def fit(self, X, y, groups=None, features=None):
        """Four steps, each of which exists for a reason worth knowing.

        1. Hold out `es_frac` for early stopping. The same slice later fits k,
           which is fine -- k is one scalar and the slice is large.

        2. n_splits-fold CV to produce OUT-OF-FOLD predictions. A GBM overfits
           its own training rows, so in-sample residuals are smaller than the
           errors the model will actually make. A variance model fitted on
           those comes out systematically too confident, and the failure looks
           like the mu+sigma*z form is wrong when it is not. The deployed mean
           model is then refit on everything -- best available mean for
           prediction, honest residuals for the variance step.

        3. Fit the variance model on log(r^2). Log, not r^2 directly: squaring
           a 22 m outlier gives 484, and an L2 fit chases it. The log
           compresses the range so the model learns typical spread rather than
           the blunders.

        4. Fit k. Step 3 estimates E[log r^2] = log(sigma^2) - 1.2704, so
           exponentiating lands 47% low on every row. k undoes it, fitted by
           matching 90% coverage on held-out data rather than using the
           theoretical exp(1.2704/2)=1.887, because that constant assumes
           gaussian residuals and these are fat-tailed.
        """
        from sklearn.model_selection import GroupKFold, KFold

        if features is not None:
            self.features_ = list(features)
        elif hasattr(X, "columns"):
            self.features_ = list(X.columns)
        X, y = _X(X, features), _y(y)

        if self.support is None:
            lo, hi = float(y.min()), float(y.max())
            m = self.support_margin * (hi - lo)
            self.support_ = (lo - m, hi + m)

        rng = np.random.default_rng(self.seed)
        es = _holdout(len(y), self.es_frac, groups, rng)
        Xf, yf = X[~es], y[~es]
        gf = None if groups is None else np.asarray(groups)[~es]
        ev = (X[es], y[es]) if es.sum() else None
        self._log(f"=== gaussian error model  (fit={(~es).sum()}, "
                  f"early-stop={es.sum()}, features={X.shape[1]}) ===")

        # 2. out-of-fold residuals
        cv = GroupKFold(self.n_splits) if gf is not None else KFold(
            self.n_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros(len(yf))
        for tr, te in cv.split(Xf, yf, gf):
            oof[te] = self._reg(Xf[tr], yf[tr], eval_set=ev).predict(Xf[te])
        resid = yf - oof

        # 1/2. deployed mean model
        self.mu_ = self._reg(Xf, yf, eval_set=ev)
        self.resid_sd_ = float(resid.std())
        self.r2_ = float(1.0 - resid.var() / yf.var())
        self._log(f"  mean model   OOF residual sd={self.resid_sd_:.4g}  "
                  f"(marginal sd={yf.std():.4g},  R^2={self.r2_:.3f})")

        # 3. log-variance model
        evv = None
        if ev is not None:
            evv = (X[es], np.log((y[es] - self.mu_.predict(X[es])) ** 2 + 1e-6))
        self.logvar_ = self._reg(Xf, np.log(resid ** 2 + 1e-6), eval_set=evv)

        # 4. bias correction
        self.k_ = 1.0
        if es.sum() > 200:
            mu_e, sg_e = self.predict_mu_sigma(X[es], _raw=True)
            self.k_ = float(np.quantile(np.abs(y[es] - mu_e) / sg_e, 0.90) / 1.6449)
            self._log(f"  sigma recal  k={self.k_:.4f}   "
                      f"(theory {np.exp(_LOG_CHI2_BIAS / 2):.3f} assuming gaussian "
                      f"residuals; lower => fatter tails than gaussian)")
            # 5. empirical shape pool, from the same held-out slice
            sg_f = sg_e * self.k_
            self._fit_pools((y[es] - mu_e) / sg_f, sg_f)
        elif self.require_recal:
            raise RecalibrationError(
                f"early-stop slice has {es.sum()} rows, need >200 to fit the sigma "
                f"bias correction. Without it sigma is ~47% too small on every row. "
                f"Raise es_frac, pass more data, or set require_recal=False and "
                f"accept a knowingly overconfident model.")
        else:
            self._log("  sigma recal  SKIPPED -- k=1.0, sigma is ~47% too small")
            if self.sampler == "empirical":
                raise RecalibrationError(
                    "sampler='empirical' needs the held-out slice (>200 rows) to "
                    "build its residual pool; none was available.")

        mu_a, sg_a = self.predict_mu_sigma(X)
        self._log(f"  sigma range  [{sg_a.min():.4g}, {sg_a.max():.4g}]  "
                  f"median={np.median(sg_a):.4g}   "
                  f"(spread {sg_a.max() / max(np.median(sg_a), 1e-9):.0f}x median "
                  f"=> heteroscedasticity learned)")
        return self

    # -- empirical shape pool ----------------------------------------
    def _fit_pools(self, z, sigma):
        """Sorted normalized residuals, optionally split into sigma bands.

        Built on the held-out slice rather than the out-of-fold training
        residuals: the variance model was fitted on those rows, so their
        sigma is in-sample and their z would come out too tame.
        """
        k = max(1, self.n_strata)
        self.z_edges_ = (np.quantile(sigma, np.linspace(0, 1, k + 1))[1:-1]
                         if k > 1 else np.array([]))
        s = self._strata(sigma)
        self.z_pools_, self.z_pgrids_ = [], []
        for i in range(k):
            p = np.sort(z[s == i])
            if len(p) < 50:
                raise RecalibrationError(
                    f"z pool stratum {i} has {len(p)} rows; lower n_strata")
            self.z_pools_.append(p)
            self.z_pgrids_.append((np.arange(1, len(p) + 1) - 0.5) / len(p))
        zs = np.sort(z)
        rb = (np.percentile(zs, 75) - np.percentile(zs, 25)) / 1.3489795
        kurt = float(((zs - zs.mean()) ** 4).mean() / zs.var() ** 2)
        self._log(f"  z pool       n={len(zs)}  strata={k}  "
                  f"robust/sd={rb / zs.std():.2f} (gaussian 1.00, uniform 1.28)  "
                  f"kurtosis={kurt:.1f} (gaussian 3)  range [{zs[0]:.2f}, {zs[-1]:.2f}]")

    def _strata(self, sigma):
        if self.z_edges_ is None or len(self.z_edges_) == 0:
            return np.zeros(len(sigma), int)
        return np.digitize(sigma, self.z_edges_)

    def _z_ppf(self, U, sigma):
        """Empirical quantile function of z, per row's sigma band. U is (n, d)."""
        s = self._strata(sigma)
        Z = np.empty_like(U, dtype=float)
        for i, (pool, pg) in enumerate(zip(self.z_pools_, self.z_pgrids_)):
            m = s == i
            if m.any():
                Z[m] = np.interp(U[m], pg, pool)
        return Z

    def _z_cdf(self, z, sigma):
        """Empirical CDF of z, per row's sigma band -- exact inverse of _z_ppf."""
        s = self._strata(sigma)
        u = np.empty_like(z, dtype=float)
        for i, (pool, pg) in enumerate(zip(self.z_pools_, self.z_pgrids_)):
            m = s == i
            if m.any():
                u[m] = np.interp(z[m], pool, pg, left=0.0, right=1.0)
        return u

    def _which(self, sampler):
        s = self.sampler if sampler is None else sampler
        if s == "empirical" and self.z_pools_ is None:
            raise RecalibrationError("empirical pool not fitted")
        return s

    # -- predict ------------------------------------------------------
    def predict_mu_sigma(self, X, _raw=False):
        """(mu, sigma) per row. sigma already includes k -- do not inflate it
        again downstream."""
        X = _X(X, self.features_)
        mu = self.mu_.predict(X)
        sigma = np.sqrt(np.exp(np.clip(self.logvar_.predict(X), -20, 20)))
        if not _raw:
            if self.k_ is None:
                raise RecalibrationError("model is not fitted")
            sigma = sigma * self.k_
        return mu, sigma

    def predict_quantiles(self, X, taus=(0.05, 0.5, 0.95), sampler=None):
        from scipy.stats import norm
        mu, sigma = self.predict_mu_sigma(X)
        t = np.asarray(taus, float)
        if self._which(sampler) == "empirical":
            Z = self._z_ppf(np.broadcast_to(t, (len(mu), len(t))).copy(), sigma)
        else:
            Z = norm.ppf(t)[None, :]
        return self._clamp(mu[:, None] + sigma[:, None] * Z)

    def sample(self, X, n_draws=1, rng=None, sampler=None):
        """Simulated range errors: mu + sigma*z, clamped to the support.
        z is N(0,1) or resampled from the held-out pool, per `sampler`.

        The clamp matters for a simulator. Unclamped the gaussian sampler
        emits values it has never seen (+29 m against an observed max of
        22.1), because a gaussian tail has no upper bound.
        """
        rng = rng or np.random.default_rng(0)
        mu, sigma = self.predict_mu_sigma(X)
        if self._which(sampler) == "empirical":
            Z = self._z_ppf(rng.random((len(mu), n_draws)), sigma)
        else:
            Z = rng.standard_normal((len(mu), n_draws))
        return self._clamp(mu[:, None] + sigma[:, None] * Z)

    def _clamp(self, v):
        if self.support_ is None:
            return v
        return np.clip(v, self.support_[0], self.support_[1])

    def cdf(self, y, X, sampler=None):
        """PIT value: the predicted CDF evaluated at the observed y. Uniform
        over [0,1] if the model is calibrated. Exact inverse of `sample` for
        the chosen sampler, so the PIT tests the thing you draw from."""
        from scipy.stats import norm
        mu, sigma = self.predict_mu_sigma(X)
        if self._which(sampler) == "empirical":
            return self._z_cdf((_y(y) - mu) / sigma, sigma)
        return norm.cdf(_y(y), mu, sigma)

    def z(self, y, X):
        """Normalized residual (y - mu)/sigma. Should be ~N(0,1) marginally AND
        within every stratum -- the single most useful diagnostic here, because
        it separates 'mu is wrong' (z off-centre) from 'sigma is wrong'
        (z sd != 1) from 'the gaussian shape is wrong' (z heavy-tailed)."""
        mu, sigma = self.predict_mu_sigma(X)
        return (_y(y) - mu) / sigma

    # -- persistence --------------------------------------------------
    def save(self, path):
        import joblib
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path):
        import joblib
        m = joblib.load(path)
        if getattr(m, "k_", None) is None:
            raise RecalibrationError(f"{path} holds an unfitted model")
        return m