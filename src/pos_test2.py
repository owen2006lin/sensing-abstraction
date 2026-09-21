"""
Conditional range-error sampler for EKF gate / positioning-error simulation.

Purpose
-------
Generate synthetic range errors that reproduce the noise in the real dataset,
so that a gate threshold and the resulting position error can be studied in
Monte Carlo without collecting more data.

This is a SIMULATION model, not a point-prediction model. The deliverable is
distributional fidelity, not sharpness -- so PIT and the gate tests below are
the metrics that matter, and CRPS is only a sanity floor.

What changed vs the GPD version
-------------------------------
  * GPD tails -> EMPIRICAL peaks-over-threshold. The old parametric tail had an
    unidentified shape (MLE wanted c=1.27 off 63 exceedances, clipped to 0.5)
    and emitted draws at +-45 m against an observed range of [-13, 22.1].
    For a simulator you do not want to extrapolate past the observed support;
    you want to reproduce it. The empirical pool does exactly that, is hard
    bounded by construction, and has no free parameters to misfit.
  * Handoff moved 0.99 -> 0.95 and calib_frac 0.15 -> 0.25, taking the
    exceedance pool from ~63 to ~450 per side.
  * tau=0.005/0.995 dropped: at those levels tau_params forced num_leaves=6,
    so the models were near-constant. The empirical pool covers that region
    better than a degenerate tree does.
  * Added a gate test suite (see `gate_report`) that scores the thing you
    actually consume: rejection rate, the distribution of errors that survive
    the gate, outage rate, and a position-error proxy.

Assumes rows are i.i.d. (confirmed) -- no block bootstrap, no group handling
in the sampler.

Drop-in for:
    from .quantile_error_model import main
    main(X_train, y_train, X_val, y_val, FEATURES)

GPU: LightGBM's CUDA build does not implement every objective. Each fit tries
CUDA first and transparently falls back to CPU, printing one warning.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; remove if you want interactive windows
import matplotlib.pyplot as plt

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None


# ======================================================================
# palette / style  (validated: lightness band, chroma, CVD separation,
# normal-vision separation and >=3:1 contrast on a #fcfcfb surface)
# ======================================================================
C_ACTUAL = "#eb6834"   # orange  - observed / real data
C_MODEL = "#2a78d6"    # blue    - quantile model
C_BASE = "#12946a"     # green   - gaussian baseline
C_REF = "#898781"      # muted   - reference / nominal
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
SEQ = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#184f95"]  # blue sequential

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": C_AXIS,
    "axes.labelcolor": C_INK2,
    "axes.titlecolor": C_INK,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.8,
    "xtick.color": C_REF,
    "ytick.color": C_REF,
    "font.size": 9,
    "axes.titlesize": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})


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


# ======================================================================
# level grid + tau-aware hyperparameters
# ======================================================================
# Handoff to the empirical tail is at 0.05 / 0.95, so levels outside that are
# carried by the resampled pool, not by a tree. 0.01 and 0.99 are kept only so
# the diagnostics have something to report at those levels -- the sampler does
# not use them. 0.005/0.995 are gone: tau_params drove them to num_leaves=6.
TAUS = np.array([
    0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 0.95, 0.99,
])

def taus_for_gate(y, thresholds, base=TAUS, lo=0.02, hi=0.98, bracket=0.03,
                  min_gap=0.015):
    """Insert grid levels where the gate thresholds actually sit.

    Between grid levels the quantile function is interpolated LINEARLY, which
    flattens curvature and biases the implied density. That matters little
    anywhere except at the gate, where the local density IS the rejection rate.

    Measured effect on a synthetic run: the T2 rejection rates barely moved
    (0.19525 -> 0.19444 at g=1.0), while the model count went 13 -> 28 and the
    pre-sort crossing rate went 0.55% -> 13.2%. So this is OFF by default in
    `main`. Turn it on and compare T2 with and against -- if your misses are
    interpolation error it will show up here, and if they do not move, the
    error is in the fit and a denser grid will not rescue it.

    A rising crossing rate is expected when levels sit closer together than the
    fitting noise; rearrangement absorbs it, but it is the signal that the grid
    has outrun what the data can resolve. min_gap keeps adjacent levels apart.

    Levels beyond [lo, hi] are dropped: out there the empirical pool governs,
    not the grid, so extra tree models would add fitting noise and nothing else.
    """
    y = _y(y)
    extra = []
    for g in np.atleast_1d(thresholds).astype(float):
        for v in (-g, g):
            p = float(np.mean(y <= v))
            extra += [p - bracket, p, p + bracket]
    t = np.array(sorted(set(np.round(
        [v for v in list(base) + extra if lo <= v <= hi], 4))))
    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] >= min_gap:
            keep.append(i)
    return t[keep]


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

_GPU_WARNED = False


def _holdout(n: int, frac: float, groups=None, rng=None) -> np.ndarray:
    """Boolean mask of length n. Holds out whole groups when groups is given."""
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


def tau_params(tau: float, n: int, base=BASE_PARAMS, min_exceed=10) -> dict:
    """Scale leaf size with tau so extreme levels have exceedances in their leaves.

    Require min_exceed points on the short side of every leaf:
      min(tau, 1-tau) * n_leaf >= min_exceed
        tau=0.5  -> ~20 (unchanged)
        tau=0.95 -> 200
        tau=0.99 -> 1000
    Then cap num_leaves so the tree can still be grown at all.
    """
    p = dict(base)
    side = min(tau, 1.0 - tau)
    mcs = int(np.ceil(min_exceed / side))
    mcs = max(int(base.get("min_child_samples", 20)), mcs)
    mcs = min(mcs, max(50, n // 8))
    p["min_child_samples"] = mcs
    p["num_leaves"] = int(max(4, min(base.get("num_leaves", 31), n / (2 * mcs))))
    return p


def _fit_one(Xtr, ytr, tau, n, use_gpu=True, seed=0, base=BASE_PARAMS, eval_set=None):
    """One quantile model. Tries CUDA, falls back to CPU on any failure.

    Early stopping is not optional: an overfit quantile model produces intervals
    that are too NARROW, which in a simulator shows up as under-dispersed noise
    and an optimistic gate rejection rate.
    """
    global _GPU_WARNED
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    p = tau_params(tau, n, base)
    p["random_state"] = seed
    kw = {}
    if eval_set is not None:
        kw = dict(eval_set=[eval_set],
                  callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])

    if use_gpu:
        try:
            m = LGBMRegressor(objective="quantile", alpha=tau, **p, **GPU_PARAMS)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return m.fit(Xtr, ytr, **kw)
        except Exception as e:  # noqa: BLE001
            if not _GPU_WARNED:
                print(f"  [gpu] CUDA fit failed ({type(e).__name__}: {e}); using CPU")
                _GPU_WARNED = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LGBMRegressor(objective="quantile", alpha=tau, **p).fit(Xtr, ytr, **kw)


# ======================================================================
# empirical peaks-over-threshold tail
# ======================================================================
class EmpiricalTail:
    """Resampled exceedances over a conditional threshold.

    Given the model's threshold quantile q(x) and the row's local scale s(x),
    the standardized excess  (y - q(x)) / s(x)  is pooled across rows and then
    resampled. The standardization is what makes pooling legitimate -- a bulk
    row exceeds q95 by 0.1 m while an NLOS row exceeds it by 8 m, and a single
    pooled distribution of RAW excesses would be meaningless.

    No free parameters, so nothing to misfit when exceedances are scarce -- the
    failure mode of the GPD version, whose MLE wanted c=1.27 off 63 points.

    CAUTION -- the bound is per row, not global. A draw is bounded by
        q(x) + s(x) * max(pool),
    and the pool max is typically set by a row with a SMALL s(x). Applied to a
    row with a large s(x) it multiplies out to a value far outside anything
    observed: on a test run this produced draws at +139 m against an observed
    max of 22. That is what `QuantileGBM.support_` exists to stop. The clamp is
    not cosmetic -- if it bites often, the standardization is mismatched and the
    tail is being applied to rows it was not estimated from.

    `ppf` and `cdf` are exact inverses on the interior (both use the same
    plotting-position grid), which is what keeps the PIT an honest test of the
    sampler rather than of a different object.
    """

    def __init__(self, pool=None, rate=None, n=0):
        self.pool = pool
        self.rate = rate
        self.n = n
        self.p_grid = None if pool is None else (np.arange(1, len(pool) + 1) - 0.5) / len(pool)

    @property
    def ok(self):
        return self.pool is not None and len(self.pool) >= 50

    @classmethod
    def fit(cls, excess: np.ndarray, n_total: int, label=""):
        """`excess` must already be STANDARDIZED by the row's local scale."""
        excess = np.sort(excess[excess > 0])
        k = len(excess)
        if k < 50:
            print(f"  [tail:{label}] only {k} exceedances -- tail disabled, grid will "
                  f"clamp. Raise calib_frac or move tail_from inward.")
            return cls(n=k)
        q = np.quantile(excess, [0.5, 0.9, 0.99])
        dyn = excess[-1] / max(q[0], 1e-9)
        flag = ""
        if dyn > 100:
            flag = (f"\n      ^^ dynamic range {dyn:.0f}x -- the standardization is NOT "
                    f"removing the heteroscedasticity.\n         One pooled shape is being "
                    f"asked to serve rows whose excesses differ by that factor.\n         "
                    f"Move `scale_from` outward (a larger, blunder-matched local scale) "
                    f"or the handoff\n         outward, and watch this number. Below ~50 "
                    f"is healthy.")
        print(f"  [tail:{label}] k={k} empirical pool  "
              f"median={q[0]:.3g}  p90={q[1]:.3g}  p99={q[2]:.3g}  max={excess[-1]:.3g}"
              f"  (dynamic range {dyn:.0f}x){flag}")
        return cls(excess, k / max(n_total, 1), k)

    def ppf(self, p):
        return np.interp(np.clip(p, 0.0, 1.0), self.p_grid, self.pool)

    def cdf(self, x):
        return np.clip(np.interp(np.maximum(x, 0.0), self.pool, self.p_grid,
                                 left=0.0, right=1.0), 0.0, 1.0)


# ======================================================================
# the model
# ======================================================================
class QuantileGBM:
    """Conditional distribution as a grid of quantiles + empirical tails."""

    def __init__(self, taus=TAUS, use_gpu=True, seed=0, base_params=BASE_PARAMS,
                 tail_from=(0.05, 0.95), scale_from=None, recalibrate=True,
                 support=None, support_margin=0.0):
        """tail_from: the (lower, upper) levels where the empirical tail takes
        over from the grid. 0.95 on a 9k calibration slice gives ~450
        exceedances per side -- dense enough to resample without visible
        repeats. Pushing the handoff further out buys a smoother grid at the
        cost of a sparser pool, and the pool is what the simulator draws from.

        scale_from: the (lower, upper) levels defining the per-row local scale
        that exceedances are divided by, DECOUPLED from the handoff. Default
        None uses the outermost grid levels.

        Why they are separate knobs. The handoff wants to be inward, where
        there are enough exceedances to resample. The scale wants to be
        outward, where the local spread reflects blunder magnitude rather than
        bulk width. Tie them together and you get the failure this was written
        to fix: handing off at 0.95, `q95 - q50` on a clean row is the width of
        the noise floor (~0.1 m), so a 10 m blunder on that row standardizes to
        ~100, while the same blunder on an already-wide row standardizes to ~1.
        Pooling those is exactly the heterogeneity the standardization exists to
        remove -- observed as a 310x dynamic range in the fitted pool. Using
        `q99 - q50` as the scale keeps a dense pool AND a scale that tracks the
        thing being scaled.

        support: (lo, hi) hard bound on every draw and every reported quantile,
        in metres. Default None learns it from the training labels, widened by
        `support_margin` (a fraction of the observed range). Pass an explicit
        pair when the physics gives you a better bound than the sample does --
        a ranging timeout, a gating threshold inside the radio, a maximum
        resolvable NLOS excess. The clamp rate is reported at sample time; a
        rate materially above the tail rate means the per-row rescaling is
        pushing the pool somewhere it was not estimated from.
        """
        self.taus = np.asarray(taus, dtype=float)
        self.use_gpu = use_gpu
        self.seed = seed
        self.base_params = base_params
        self.recalibrate = recalibrate
        self.support = support
        self.support_margin = support_margin
        self.support_ = support
        self.clamp_rate_ = np.nan
        self.recal_ = None
        self.models_ = {}
        self.hi_ = EmpiricalTail()
        self.lo_ = EmpiricalTail()
        self.lo_i = int(np.argmin(np.abs(self.taus - tail_from[0])))
        self.hi_i = int(np.argmin(np.abs(self.taus - tail_from[1])))
        self.i50 = int(np.argmin(np.abs(self.taus - 0.5)))
        sf = (self.taus[0], self.taus[-1]) if scale_from is None else scale_from
        self.scale_from = sf
        self.sc_lo_i = int(np.argmin(np.abs(self.taus - sf[0])))
        self.sc_hi_i = int(np.argmin(np.abs(self.taus - sf[1])))
        self.crossing_rate_ = np.nan

    def _scales(self, Q):
        """Per-row local scale for each tail. Exceedances are divided by these
        before pooling, so one pool serves rows whose spread differs by orders
        of magnitude -- provided the scale tracks the excesses. See
        `scale_from`: when it does not, the pool's dynamic range blows up and
        the tail stops being conditional in any useful sense."""
        eps = 1e-9
        s_hi = np.maximum(Q[:, self.sc_hi_i] - Q[:, self.i50], eps)
        s_lo = np.maximum(Q[:, self.i50] - Q[:, self.sc_lo_i], eps)
        return s_lo, s_hi

    # -- fit ----------------------------------------------------------
    def fit(self, X, y, calib_frac=0.25, es_frac=0.15, groups=None):
        """Three-way split: fit / early-stop / calibrate.

        calib must be out-of-sample or the exceedance pool is biased small by
        the models' own overfit and the simulated tail comes out too thin. The
        early-stopping slice is kept separate so neither job contaminates the
        other.
        """
        X, y = _X(X), _y(y)
        if self.support is None:
            lo, hi = float(y.min()), float(y.max())
            m = self.support_margin * (hi - lo)
            self.support_ = (lo - m, hi + m)
        rng = np.random.default_rng(self.seed)
        cal = _holdout(len(y), calib_frac, groups, rng)
        rest = ~cal
        es_rel = _holdout(int(rest.sum()), es_frac,
                          None if groups is None else np.asarray(groups)[rest], rng)
        es = np.zeros(len(y), bool)
        es[np.flatnonzero(rest)[es_rel]] = True
        fit = rest & ~es

        Xf, yf = X[fit], y[fit]
        ev = (X[es], y[es]) if es.sum() else None
        print(f"\n=== fitting {len(self.taus)} quantile models "
              f"(fit={fit.sum()}, early-stop={es.sum()}, calib={cal.sum()}) ===")
        for t in self.taus:
            p = tau_params(t, len(yf), self.base_params)
            self.models_[t] = _fit_one(Xf, yf, t, len(yf), self.use_gpu,
                                       self.seed, self.base_params, eval_set=ev)
            ni = getattr(self.models_[t], "best_iteration_", None) or \
                self.models_[t].n_estimators
            print(f"  tau={t:<6.3f} min_child_samples={p['min_child_samples']:<5d} "
                  f"num_leaves={p['num_leaves']:<4d} trees={ni}")

        if cal.sum():
            yc = y[cal]
            Q = self._grid(X[cal])
            s_lo, s_hi = self._scales(Q)
            self.hi_ = EmpiricalTail.fit((yc - Q[:, self.hi_i]) / s_hi, len(yc),
                                         f"upper>q{self.taus[self.hi_i]}")
            self.lo_ = EmpiricalTail.fit((Q[:, self.lo_i] - yc) / s_lo, len(yc),
                                         f"lower<q{self.taus[self.lo_i]}")
            if self.recalibrate:
                self.recal_ = np.sort(self._cdf_raw(yc, Q=Q))
                ks = np.max(np.abs(self.recal_ - np.linspace(0, 1, len(self.recal_))))
                print(f"  [recal] raw PIT KS on calib = {ks:.4f} -> remapped to uniform"
                      f"  (marginal only; does not touch conditional coverage)")
        return self

    # -- predict ------------------------------------------------------
    def _grid(self, X):
        """Raw sorted quantile grid. Sorting is the Chernozhukov rearrangement:
        provably weakly closer to the true quantile function, never worse."""
        X = _X(X)
        Q = np.column_stack([self.models_[t].predict(X) for t in self.taus])
        self.crossing_rate_ = float(np.mean(np.diff(Q, axis=1) < 0))
        return np.sort(Q, axis=1)

    def predict_quantiles(self, X=None, Q=None, taus=None):
        """(n, K) quantile matrix of the SAMPLER's distribution -- recalibration
        and empirical tails included. The metrics therefore grade the same
        object the simulator draws from; otherwise coverage and PIT would score
        a model you do not ship."""
        Q = self._grid(X) if Q is None else Q
        t = self.taus if taus is None else np.asarray(taus, float)
        U = np.broadcast_to(self._G_inv(t), (Q.shape[0], len(t)))
        return self._eval(Q, np.ascontiguousarray(U))

    # -- inverse-CDF sampling ----------------------------------------
    def sample(self, X=None, Q=None, n_draws=1, rng=None, verbose=True):
        """Draw u ~ U(0,1), remap through the recalibration, interpolate the
        quantile grid, resample the empirical pool beyond it, clamp to support."""
        rng = rng or np.random.default_rng(0)
        Q = self._grid(X) if Q is None else Q
        S = self._eval(Q, self._G_inv(rng.random((Q.shape[0], n_draws))))
        if verbose and np.isfinite(self.clamp_rate_):
            tail_rate = (1 - self.taus[self.hi_i]) + self.taus[self.lo_i]
            msg = "" if self.clamp_rate_ < 0.2 * tail_rate else \
                "   <-- large vs tail rate: per-row rescaling is overshooting"
            print(f"  [support] clamped {self.clamp_rate_:.4%} of draws to "
                  f"[{self.support_[0]:.4g}, {self.support_[1]:.4g}]"
                  f"  (tail region is {tail_rate:.1%} of draws){msg}")
        return S

    def _eval(self, Q, U):
        """Quantile function of the RAW grid Q evaluated at raw-scale levels U."""
        t = self.taus
        idx = np.clip(np.searchsorted(t, U, side="right") - 1, 0, len(t) - 2)
        t0, t1 = t[idx], t[idx + 1]
        q0 = np.take_along_axis(Q, idx, axis=1)
        q1 = np.take_along_axis(Q, idx + 1, axis=1)
        out = q0 + (U - t0) / (t1 - t0) * (q1 - q0)

        # empirical tails take over at the handoff levels, rescaled per row
        th, tl = t[self.hi_i], t[self.lo_i]
        s_lo, s_hi = self._scales(Q)
        hi = U > th
        if hi.any() and self.hi_.ok:
            out[hi] = (np.broadcast_to(Q[:, [self.hi_i]], U.shape)[hi]
                       + np.broadcast_to(s_hi[:, None], U.shape)[hi]
                       * self.hi_.ppf((U[hi] - th) / (1.0 - th)))
        lo = U < tl
        if lo.any() and self.lo_.ok:
            out[lo] = (np.broadcast_to(Q[:, [self.lo_i]], U.shape)[lo]
                       - np.broadcast_to(s_lo[:, None], U.shape)[lo]
                       * self.lo_.ppf((tl - U[lo]) / tl))

        # Hard support clamp. The per-row rescaling of a pooled tail is not
        # bounded by the observed range on its own -- a large s(x) multiplied by
        # the pool's max standardized excess lands well outside it. For a
        # simulator that is never what you want: a draw the data never produced
        # should not be fed to the filter.
        if self.support_ is not None:
            self.clamp_rate_ = float(np.mean((out < self.support_[0]) |
                                             (out > self.support_[1])))
            np.clip(out, self.support_[0], self.support_[1], out=out)
        return out

    # -- global recalibration ----------------------------------------
    # The gaussian baseline gets one scalar fit on held-out data; this is the
    # quantile model's equivalent, and withholding it would rig the comparison
    # the other way. If the raw PIT has CDF G, then G(F(y)) is uniform by
    # construction (Kuleshov et al. 2018). Fit on calib, never on val.
    def _G_inv(self, u):
        return np.quantile(self.recal_, np.clip(u, 0, 1)) if self.recal_ is not None else u

    def _G(self, p):
        if self.recal_ is None:
            return p
        return np.searchsorted(self.recal_, p, side="right") / len(self.recal_)

    def cdf(self, y, X=None, Q=None, recal=True):
        u = self._cdf_raw(y, X=X, Q=Q)
        return np.clip(self._G(u), 0.0, 1.0) if recal else u

    def _cdf_raw(self, y, X=None, Q=None):
        """Predicted CDF evaluated at the observed y -- the PIT value.
        Exact inverse of `sample`, tails included."""
        Q = self._grid(X) if Q is None else Q
        y = _y(y)
        t = self.taus
        jit = np.arange(Q.shape[1]) * 1e-10  # break ties in flat regions
        u = np.array([np.interp(yi, Qi + jit, t) for yi, Qi in zip(y, Q)])

        th, tl = t[self.hi_i], t[self.lo_i]
        s_lo, s_hi = self._scales(Q)
        hi = y > Q[:, self.hi_i]
        if hi.any() and self.hi_.ok:
            u[hi] = th + (1 - th) * self.hi_.cdf((y[hi] - Q[hi, self.hi_i]) / s_hi[hi])
        lo = y < Q[:, self.lo_i]
        if lo.any() and self.lo_.ok:
            u[lo] = tl * (1.0 - self.lo_.cdf((Q[lo, self.lo_i] - y[lo]) / s_lo[lo]))
        return np.clip(u, 0.0, 1.0)


# ======================================================================
# baseline: the current pipeline (bias -> OOF residuals -> sigma -> mu+sigma*z)
# ======================================================================
class GaussianBaseline:
    """L2 mean model + log-variance model on OOF residuals. Same interface."""

    def __init__(self, n_splits=5, use_gpu=True, seed=0):
        self.n_splits, self.use_gpu, self.seed = n_splits, use_gpu, seed

    def _reg(self, Xtr, ytr, objective="regression", eval_set=None):
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        p = dict(BASE_PARAMS, random_state=self.seed)
        kw = {}
        if eval_set is not None:
            kw = dict(eval_set=[eval_set],
                      callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.use_gpu:
                try:
                    return LGBMRegressor(objective=objective, **p, **GPU_PARAMS).fit(
                        Xtr, ytr, **kw)
                except Exception:  # noqa: BLE001
                    pass
            return LGBMRegressor(objective=objective, **p).fit(Xtr, ytr, **kw)

    def fit(self, X, y, groups=None, es_frac=0.15):
        """Gets the same early-stopping budget and the same held-out slice as
        the quantile model, so the head-to-head is not rigged either way."""
        from sklearn.model_selection import GroupKFold, KFold
        X, y = _X(X), _y(y)
        rng = np.random.default_rng(self.seed)
        es = _holdout(len(y), es_frac, groups, rng)
        Xf, yf = X[~es], y[~es]
        gf = None if groups is None else np.asarray(groups)[~es]
        ev = (X[es], y[es]) if es.sum() else None
        print(f"\n=== fitting gaussian baseline (mu + sigma*z) "
              f"(fit={(~es).sum()}, early-stop={es.sum()}) ===")

        cv = GroupKFold(self.n_splits) if gf is not None else KFold(
            self.n_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros(len(yf))
        for tr, te in cv.split(Xf, yf, gf):
            oof[te] = self._reg(Xf[tr], yf[tr], eval_set=ev).predict(Xf[te])

        self.mu_ = self._reg(Xf, yf, eval_set=ev)
        resid = yf - oof
        evv = (X[es], np.log((y[es] - self.mu_.predict(X[es])) ** 2 + 1e-6)) if ev else None
        self.logvar_ = self._reg(Xf, np.log(resid ** 2 + 1e-6), eval_set=evv)
        print(f"  OOF residual sd={resid.std():.4g}  "
              f"(step-2 residuals; sigma fit on log of their square)")

        # Regressing log(r^2) under L2 estimates E[log r^2], and for gaussian
        # residuals E[log r^2] = log(sigma^2) - 1.2704 (mean of a log chi^2_1).
        # Exponentiating lands ~0.53x low. Recalibrate one scalar on held-out
        # data by matching 90% coverage -- robust to that bias AND to heavy
        # tails, which a moment-matched scalar would not be.
        self.k_ = 1.0
        if es.sum() > 200:
            mu_e, sg_e = self.predict_mu_sigma(X[es])
            self.k_ = float(np.quantile(np.abs(y[es] - mu_e) / sg_e, 0.90) / 1.6449)
            print(f"  sigma recalibration k={self.k_:.4f}  "
                  f"(theory says ~{np.exp(1.2704 / 2):.3f} from the log-chi^2 bias alone; "
                  f"without this the baseline is rigged to fail)")
        return self

    def predict_mu_sigma(self, X):
        X = _X(X)
        mu = self.mu_.predict(X)
        sigma = np.sqrt(np.exp(np.clip(self.logvar_.predict(X), -20, 20)))
        return mu, sigma * getattr(self, "k_", 1.0)

    def sample(self, X, n_draws=1, rng=None):
        rng = rng or np.random.default_rng(0)
        mu, sigma = self.predict_mu_sigma(X)
        return mu[:, None] + sigma[:, None] * rng.standard_normal((len(mu), n_draws))

    def cdf(self, y, X):
        from scipy.stats import norm
        mu, sigma = self.predict_mu_sigma(X)
        return norm.cdf(_y(y), mu, sigma)

    def predict_quantiles(self, X, taus=None):
        taus = TAUS if taus is None else taus
        from scipy.stats import norm
        mu, sigma = self.predict_mu_sigma(X)
        return mu[:, None] + sigma[:, None] * norm.ppf(taus)[None, :]


# ======================================================================
# distributional metrics (correctness floor, not the primary target)
# ======================================================================
def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1.0) * d))


def pinball_table(y, Q, taus=None):
    taus = TAUS if taus is None else taus
    y = _y(y)
    model = np.array([pinball(y, Q[:, i], t) for i, t in enumerate(taus)])
    base = np.array([pinball(y, np.full_like(y, np.quantile(y, t)), t) for t in taus])
    return model, base


def crps_from_samples(S, y):
    """CRPS = E|X-y| - 0.5*E|X-X'|, via the Gini mean-difference identity."""
    y = _y(y)
    m = S.shape[1]
    t1 = np.abs(S - y[:, None]).mean(axis=1)
    Ss = np.sort(S, axis=1)
    w = (2 * np.arange(1, m + 1) - m - 1)
    t2 = (2.0 / m ** 2) * (Ss * w).sum(axis=1)
    return float(np.mean(t1 - 0.5 * t2))


def coverage_table(y, Q, taus=None):
    taus = TAUS if taus is None else taus
    y = _y(y)
    return np.array([np.mean(y <= Q[:, i]) for i in range(len(taus))])


def conditional_coverage(y, Q, taus=None, lo=0.05, hi=0.95, nbins=10):
    """Marginal coverage is easy to pass by accident. Bin by predicted spread and
    check coverage inside each bin -- this is the check that discriminates."""
    taus = TAUS if taus is None else taus
    y = _y(y)
    il, ih = int(np.argmin(np.abs(taus - lo))), int(np.argmin(np.abs(taus - hi)))
    spread = Q[:, ih] - Q[:, il]
    edges = np.quantile(spread, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(spread, edges[1:-1]), 0, nbins - 1)
    cov, mid = [], []
    for b in range(nbins):
        s = idx == b
        cov.append(np.mean((y[s] >= Q[s, il]) & (y[s] <= Q[s, ih])) if s.sum() else np.nan)
        mid.append(np.median(spread[s]) if s.sum() else np.nan)
    return np.array(mid), np.array(cov), taus[ih] - taus[il]


def summarize(name, y, Q, S, pit, taus=None):
    taus = TAUS if taus is None else taus
    y = _y(y)
    pl_, base = pinball_table(y, Q, taus)
    n = len(y)
    out = dict(
        name=name,
        crps_samples=crps_from_samples(S, y),
        pinball_mean=float(pl_.mean()),
        pinball_skill=float(1 - pl_.mean() / base.mean()),
        pit_ks=float(np.max(np.abs(np.sort(pit) - np.linspace(0, 1, n)))),
        cov90=float(np.mean((y >= Q[:, np.argmin(np.abs(taus - 0.05))]) &
                            (y <= Q[:, np.argmin(np.abs(taus - 0.95))]))),
    )
    crit = 1.36 / np.sqrt(n)
    out["pit_pass"] = out["pit_ks"] < crit
    print(f"\n--- {name} ---")
    print(f"  CRPS (samples)     {out['crps_samples']:.5f}   (sharpness; secondary here)")
    print(f"  mean pinball       {out['pinball_mean']:.5f}  "
          f"(skill vs unconditional: {out['pinball_skill']:+.3%})")
    print(f"  PIT KS distance    {out['pit_ks']:.4f}   "
          f"(5% critical value {crit:.4f} -> "
          f"{'CALIBRATED' if out['pit_pass'] else 'REJECT uniformity'})")
    print(f"  90% coverage       {out['cov90']:.3%}   (nominal 90%)")
    out["draw_min"], out["draw_max"] = float(S.min()), float(S.max())
    out["frac_outside"] = float(np.mean((S < y.min()) | (S > y.max())))
    print(f"  sample range       [{out['draw_min']:.3g}, {out['draw_max']:.3g}]   "
          f"(observed [{y.min():.3g}, {y.max():.3g}];  "
          f"{out['frac_outside']:.3%} of draws outside it)")
    return out


# ======================================================================
# GATE TESTS -- the metrics that match how this model is consumed
# ======================================================================
def _replicas(S, n_rep=20):
    """Each column of S is a synthetic dataset of the same size and the same
    feature composition as the real val set -- the apples-to-apples object for a
    two-sample comparison. Statistics are computed per replica so the Monte
    Carlo spread is directly comparable to the real data's sampling error."""
    m = S.shape[1]
    idx = np.unique(np.linspace(0, m - 1, min(n_rep, m)).astype(int))
    return [S[:, i] for i in idx]


def marginal_fidelity(y, S, n_rep=20):
    """Two-sample KS and Wasserstein between a synthetic replica and the real
    data, against the 5% critical value for two samples of size n.

    This is the most direct test of 'does the sampler recreate this noise' --
    PIT tests the conditional model, this tests the thing you will actually
    feed the filter."""
    from scipy.stats import ks_2samp, wasserstein_distance
    y = _y(y)
    n = len(y)
    reps = _replicas(S, n_rep)
    ks = np.array([ks_2samp(y, r).statistic for r in reps])
    w = np.array([wasserstein_distance(y, r) for r in reps])
    crit = 1.36 * np.sqrt(2.0 / n)   # two-sample, equal sizes, alpha=0.05
    return dict(ks_mean=float(ks.mean()), ks_sd=float(ks.std()),
                ks_crit=float(crit), ks_pass=bool(ks.mean() < crit),
                w_mean=float(w.mean()), w_sd=float(w.std()),
                sd_real=float(y.std()), sd_sim=float(np.mean([r.std() for r in reps])))


def _wilson(k, n, z=1.96):
    """Wilson score interval for a binomial rate.

    The normal approximation p +- z*sqrt(p(1-p)/n) collapses to ZERO WIDTH when
    p = 0, which is common here: in the narrow-spread deciles almost nothing
    exceeds the gate. A zero-width interval marks every non-zero simulated rate
    as a miss, so the test reports failures that are artifacts of its own
    arithmetic. Wilson stays sensible at the boundary -- at k=0 it gives
    [0, ~z^2/n] rather than [0, 0].
    """
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(c - h, 0.0)), float(min(c + h, 1.0))


def gate_rates(y, S, thresholds, n_rep=20):
    """Rejection rate P(|e| > g), real vs simulated, per threshold.

    The real rate carries a binomial 95% interval; the simulated rate is
    estimated from the full draw pool so its own error is negligible. The test
    is simply whether the simulated rate lands inside the real interval -- if it
    does not, the gate will fire at the wrong frequency in simulation and every
    downstream conclusion inherits that error."""
    y = _y(y)
    n = len(y)
    pool = S.ravel()
    reps = _replicas(S, n_rep)
    rows = []
    for g in thresholds:
        k = int((np.abs(y) > g).sum())
        p = k / n
        lo, hi = _wilson(k, n)
        ps = float(np.mean(np.abs(pool) > g))
        pr = np.array([np.mean(np.abs(r) > g) for r in reps])
        rows.append(dict(g=float(g), real=p, lo=lo, hi=hi, sim=ps,
                         sim_sd=float(pr.std()), n_real=k,
                         ratio=float(ps / p) if p > 0 else np.nan,
                         ok=bool(lo <= ps <= hi)))
    return rows


def accepted_moments(y, S, thresholds, n_rep=20):
    """Moments of the errors that SURVIVE the gate.

    These, not the rejected blunders, are what corrupt the position solution.
    `bias` matters most: a skewed error distribution leaves a non-zero mean
    among accepted measurements, which the filter cannot distinguish from a
    real position offset."""
    y = _y(y)
    reps = _replicas(S, n_rep)
    rows = []
    for g in thresholds:
        a = y[np.abs(y) <= g]
        sim = [r[np.abs(r) <= g] for r in reps]
        sim = [s for s in sim if len(s) > 10]
        if len(a) < 10 or not sim:
            continue
        rows.append(dict(
            g=float(g),
            rms_real=float(np.sqrt(np.mean(a ** 2))),
            rms_sim=float(np.mean([np.sqrt(np.mean(s ** 2)) for s in sim])),
            rms_sim_sd=float(np.std([np.sqrt(np.mean(s ** 2)) for s in sim])),
            bias_real=float(a.mean()),
            bias_sim=float(np.mean([s.mean() for s in sim])),
            bias_sim_sd=float(np.std([s.mean() for s in sim])),
            sd_real=float(a.std()), sd_sim=float(np.mean([s.std() for s in sim])),
        ))
    return rows


def gate_by_stratum(y, S, spread, g, nbins=10, n_rep=20):
    """Rejection rate within predicted-spread deciles.

    The marginal rate can match by accident -- too many blunders in the easy
    rows cancelling too few in the hard rows. This is the conditional version,
    and it is the one that discriminates. For a gate study it matters because
    rejections are not spread evenly across the geometry: if the model puts them
    in the wrong rows, the simulated outage pattern is wrong even when the
    simulated outage RATE is right."""
    y = _y(y)
    edges = np.quantile(spread, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(spread, edges[1:-1]), 0, nbins - 1)
    reps = _replicas(S, n_rep)
    real, sim, lo, hi = [], [], [], []
    for b in range(nbins):
        m = idx == b
        nb = int(m.sum())
        k = int((np.abs(y[m]) > g).sum())
        a, c = _wilson(k, nb)
        real.append(k / nb if nb else np.nan)
        lo.append(a)
        hi.append(c)
        sim.append(float(np.mean([np.mean(np.abs(r[m]) > g) for r in reps])) if nb else np.nan)
    return np.array(real), np.array(sim), np.array(lo), np.array(hi)


def position_proxy(e_pool, g, n_anchors=4, n_trials=200_000, rng=None):
    """First-order stand-in for the position error the filter experiences.

    Draw n_anchors i.i.d. range errors, apply the gate, average the survivors.
    Under isotropic geometry with DOP=1 the averaged residual is proportional to
    the position error, so comparing this distribution between real and
    simulated noise tests the end-to-end quantity rather than the marginal.

    Returns (proxy errors, outage rate). Outage = every anchor rejected, which
    is a failure mode the gate threshold directly controls and which a marginal
    comparison cannot see at all.
    """
    rng = rng or np.random.default_rng(0)
    e_pool = np.asarray(e_pool, float)
    idx = rng.integers(0, len(e_pool), size=(n_trials, n_anchors))
    E = e_pool[idx]
    keep = np.abs(E) <= g
    nk = keep.sum(axis=1)
    ok = nk > 0
    est = np.divide((E * keep).sum(axis=1), np.maximum(nk, 1))
    return est[ok], float(np.mean(~ok))


def gate_report(y, S_q, S_g, spread, thresholds, operating_gate, n_anchors=4,
                rng=None, n_rep=20, handoff=None):
    """Print the full gate-oriented comparison and return everything computed."""
    y = _y(y)
    rng = rng or np.random.default_rng(0)
    print("\n" + "=" * 78)
    print(f"EKF GATE DIAGNOSTICS   (val n={len(y)}, {S_q.shape[1]} draws/row, "
          f"{n_rep} synthetic replicas)")
    print("=" * 78)

    if handoff is not None:
        tl, th = handoff
        pu, pl_ = float(np.mean(y <= operating_gate)), float(np.mean(y <= -operating_gate))
        where = lambda p: "empirical tail" if (p > th or p < tl) else "quantile grid"
        print(f"\n-- T0. Which component governs the gate at g={operating_gate} m?")
        print(f"   +{operating_gate} m sits at marginal tau={pu:.4f} -> {where(pu)}")
        print(f"   -{operating_gate} m sits at marginal tau={pl_:.4f} -> {where(pl_)}")
        print(f"   (handoff at tau={tl} / {th}. If T2 misses at this gate, this "
              f"line tells you\n    which piece to tune: the tree grid or the "
              f"exceedance pool.)")

    mf_q = marginal_fidelity(y, S_q, n_rep)
    mf_g = marginal_fidelity(y, S_g, n_rep)
    print("\n-- T1. Marginal fidelity: does a synthetic dataset look like the real one?")
    print(f"   two-sample KS 5% critical value = {mf_q['ks_crit']:.4f}")
    for nm, mf in (("quantile", mf_q), ("gaussian", mf_g)):
        print(f"   {nm:<9} KS={mf['ks_mean']:.4f} (+-{mf['ks_sd']:.4f})  "
              f"W1={mf['w_mean']:.4f}  sd={mf['sd_sim']:.4f} vs real {mf['sd_real']:.4f}"
              f"   {'PASS' if mf['ks_pass'] else 'FAIL'}")

    gr_q = gate_rates(y, S_q, thresholds, n_rep)
    gr_g = gate_rates(y, S_g, thresholds, n_rep)
    print("\n-- T2. Gate rejection rate P(|e| > g): does the gate fire as often?")
    print(f"   {'gate(m)':>8} {'real':>9} {'95% CI (Wilson)':>19} {'n_evt':>6} "
          f"{'qgbm':>9} {'x':>6} {'':>5} {'gauss':>9} {'x':>6}")
    for a, b in zip(gr_q, gr_g):
        print(f"   {a['g']:>8.2f} {a['real']:>9.5f} "
              f"[{a['lo']:.5f},{a['hi']:.5f}] {a['n_real']:>6d} "
              f"{a['sim']:>9.5f} {a['ratio']:>6.2f} {'ok' if a['ok'] else 'MISS':>5} "
              f"{b['sim']:>9.5f} {b['ratio']:>6.2f} {'ok' if b['ok'] else 'MISS'}")
    print("   (x = sim/real. Read the ratio, not just ok/MISS: at large n_evt the "
          "CI is tight\n    enough that a 5% error is a MISS, and 1.05x may be fine "
          "for your purpose.)")

    am_q = accepted_moments(y, S_q, thresholds, n_rep)
    am_g = accepted_moments(y, S_g, thresholds, n_rep)
    print("\n-- T3. Errors that SURVIVE the gate (what actually corrupts position)")
    print(f"   {'gate(m)':>8} {'RMS real':>9} {'RMS qgbm':>9} {'RMS gauss':>10} "
          f"{'bias real':>10} {'bias qgbm':>10} {'bias gauss':>11}")
    for a, b in zip(am_q, am_g):
        print(f"   {a['g']:>8.2f} {a['rms_real']:>9.4f} {a['rms_sim']:>9.4f} "
              f"{b['rms_sim']:>10.4f} {a['bias_real']:>10.4f} {a['bias_sim']:>10.4f} "
              f"{b['bias_sim']:>11.4f}")

    cr_q, cs_q, clo, chi = gate_by_stratum(y, S_q, spread, operating_gate, n_rep=n_rep)
    cr_g, cs_g, _, _ = gate_by_stratum(y, S_g, spread, operating_gate, n_rep=n_rep)
    miss_q = int(np.sum((cs_q < clo) | (cs_q > chi)))
    miss_g = int(np.sum((cs_g < clo) | (cs_g > chi)))
    print(f"\n-- T4. Conditional rejection rate by predicted-spread decile "
          f"(g={operating_gate} m)")
    print(f"   deciles outside the real 95% CI:  quantile {miss_q}/10   "
          f"gaussian {miss_g}/10   (0-1 is healthy)")

    pr, out_r = position_proxy(y, operating_gate, n_anchors, rng=rng)
    pq, out_q = position_proxy(S_q.ravel(), operating_gate, n_anchors, rng=rng)
    pg, out_g = position_proxy(S_g.ravel(), operating_gate, n_anchors, rng=rng)
    f = lambda v: (float(np.sqrt(np.mean(v ** 2))), float(np.mean(v)),
                   float(np.percentile(np.abs(v), 95)))
    print(f"\n-- T5. Position-error proxy: mean of gated errors over "
          f"{n_anchors} anchors (g={operating_gate} m)")
    print(f"   {'source':<10} {'RMS':>9} {'bias':>9} {'P95|e|':>9} {'outage':>9}")
    for nm, v, o in (("real", pr, out_r), ("quantile", pq, out_q), ("gaussian", pg, out_g)):
        r, m_, p95 = f(v)
        print(f"   {nm:<10} {r:>9.4f} {m_:>9.4f} {p95:>9.4f} {o:>9.5f}")
    print("   (outage = all anchors rejected; a failure mode a marginal test cannot see)")
    print("=" * 78)

    return dict(marginal=dict(quantile=mf_q, gaussian=mf_g),
                rates=dict(quantile=gr_q, gaussian=gr_g),
                accepted=dict(quantile=am_q, gaussian=am_g),
                stratum=dict(real=cr_q, lo=clo, hi=chi, quantile=cs_q, gaussian=cs_g),
                proxy=dict(real=(pr, out_r), quantile=(pq, out_q), gaussian=(pg, out_g)))


# ======================================================================
# plots
# ======================================================================
def _finish(ax, title, xlabel=None, ylabel=None, legend=True):
    ax.set_title(title, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)


def plot_distributions(y_val, S_q, Q_q, S_g, outdir):
    y = _y(y_val)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    lo, hi = np.quantile(y, [0.002, 0.995])
    bins = np.linspace(lo, hi, 120)
    for v, c, nm in ((y, C_ACTUAL, "actual"), (S_q[:, 0], C_MODEL, "sampled")):
        ax.hist(v[(v >= lo) & (v <= hi)], bins=bins, color=c, alpha=0.6, density=True,
                label=f"{nm}  ({np.mean((v < lo) | (v > hi)):.2%} off-axis)")
    _finish(ax, "1. Marginal check — one synthetic replica vs actual\n"
                "same n, same feature mix; these SHOULD overlap",
            "range error (m)", "density")

    ax = axes[0, 1]
    ts = np.linspace(0, np.quantile(y, 0.9999), 200)
    surv = lambda v: [(np.abs(v) > t).mean() for t in ts]
    ax.semilogy(ts, surv(y), color=C_ACTUAL, label="actual")
    ax.semilogy(ts, surv(S_q.ravel()), color=C_MODEL, ls="-", label="quantile + empirical tail")
    ax.semilogy(ts, surv(S_g.ravel()), color=C_BASE, ls="--", label="gaussian mu+sigma*z")
    ax.set_ylim(1e-5, 1.5)
    _finish(ax, "2. Gate curve — P(|e| > t), log scale\n"
                "this IS the rejection-rate curve; where it misses, the gate misfires",
            "gate threshold t (m)", "P(|e| > t)")

    ax = axes[1, 0]
    q = np.linspace(0.001, 0.999, 400)
    ax.plot(np.quantile(y, q), np.quantile(S_q.ravel(), q), "o", ms=3,
            color=C_MODEL, label="quantile model")
    ax.plot(np.quantile(y, q), np.quantile(S_g.ravel(), q), "s", ms=3,
            color=C_BASE, label="gaussian")
    lim = [min(y.min(), 0), y.max()]
    ax.plot(lim, lim, color=C_REF, ls="--", lw=1.5, label="y = x")
    _finish(ax, "3. Q–Q, simulated vs actual\n"
                "departures at the ends are what the gate sees",
            "actual quantile (m)", "simulated quantile (m)")

    ax = axes[1, 1]
    i05, i25, i50 = (np.argmin(np.abs(TAUS - t)) for t in (0.05, 0.30, 0.50))
    i75, i95 = (np.argmin(np.abs(TAUS - t)) for t in (0.70, 0.95))
    order = np.argsort(Q_q[:, i95] - Q_q[:, i05])
    nb = 150
    grp = np.array_split(order, nb)
    xs = np.arange(nb)
    agg = lambda i: np.array([Q_q[g, i].mean() for g in grp])
    for a, b, col in ((i05, i95, SEQ[1]), (i25, i75, SEQ[3])):
        ax.fill_between(xs, agg(a), agg(b), color=col, alpha=0.75, linewidth=0,
                        label=f"{round((TAUS[b] - TAUS[a]) * 100)}% interval")
    ax.plot(xs, agg(i50), color=SEQ[4], lw=1.8, label="median")
    ax.scatter(xs, [np.median(y[g]) for g in grp], s=9, color=C_ACTUAL,
               alpha=0.85, label="actual median", zorder=3)
    _finish(ax, "4. Predicted intervals, rows grouped by width\n"
                "band should widen left→right and track the dots",
            "val rows (binned, narrow → wide)", "range error (m)")

    fig.tight_layout()
    p = os.path.join(outdir, "01_distributions.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_calibration(y_val, Q_q, pit_q, Q_g, pit_g, outdir):
    y = _y(y_val)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, pit, col, nm, num in ((axes[0, 0], pit_q, C_MODEL, "quantile model", 5),
                                  (axes[0, 1], pit_g, C_BASE, "gaussian mu+sigma*z", 6)):
        ax.hist(pit, bins=25, color=col, alpha=0.8, density=True, label=nm)
        ax.axhline(1.0, color=C_REF, ls="--", lw=1.5, label="uniform (calibrated)")
        ax.set_ylim(0, max(2.5, np.histogram(pit, 25, density=True)[0].max() * 1.15))
        _finish(ax, f"{num}. PIT — {nm}\nU-shape = too narrow, hump = too wide",
                "PIT value", "density")

    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], color=C_REF, ls="--", lw=1.5, label="nominal")
    ax.plot(TAUS, coverage_table(y, Q_q), "o-", color=C_MODEL, ms=6, label="quantile model")
    ax.plot(TAUS, coverage_table(y, Q_g), "s--", color=C_BASE, ms=6, label="gaussian")
    _finish(ax, "7. Coverage — empirical P(y ≤ q_τ) vs τ", "nominal τ", "empirical")

    ax = axes[1, 1]
    mq, cq, nom = conditional_coverage(y, Q_q)
    mg, cg, _ = conditional_coverage(y, Q_g)
    ax.axhline(nom, color=C_REF, ls="--", lw=1.5, label=f"nominal {nom:.0%}")
    ax.plot(range(10), cq, "o-", color=C_MODEL, ms=6, label="quantile model")
    ax.plot(range(10), cg, "s--", color=C_BASE, ms=6, label="gaussian")
    ax.set_ylim(0, 1.05)
    _finish(ax, "8. Conditional coverage by predicted-spread decile\n"
                "flat on nominal = right for the right reason",
            "spread decile (narrow → wide)", "90% coverage")

    fig.tight_layout()
    p = os.path.join(outdir, "02_calibration.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_gate(y_val, rep, thresholds, operating_gate, outdir):
    y = _y(y_val)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    g = np.array([r["g"] for r in rep["rates"]["quantile"]])

    ax = axes[0, 0]
    rq = rep["rates"]["quantile"]
    ax.fill_between(g, [r["lo"] for r in rq], [r["hi"] for r in rq],
                    color=C_ACTUAL, alpha=0.2, linewidth=0, label="real 95% CI")
    ax.semilogy(g, [r["real"] for r in rq], "-", color=C_ACTUAL, label="real")
    ax.semilogy(g, [r["sim"] for r in rq], "o-", color=C_MODEL, ms=6, label="quantile model")
    ax.semilogy(g, [r["sim"] for r in rep["rates"]["gaussian"]], "s--",
                color=C_BASE, ms=6, label="gaussian")
    ax.axvline(operating_gate, color=C_REF, ls=":", lw=1.5)
    _finish(ax, "T2. Gate rejection rate vs threshold\n"
                "sim inside the band = the gate fires as often as in reality",
            "gate threshold (m)", "P(|e| > g)")

    ax = axes[0, 1]
    aq, ag = rep["accepted"]["quantile"], rep["accepted"]["gaussian"]
    ga = np.array([r["g"] for r in aq])
    ax.plot(ga, [r["rms_real"] for r in aq], "-", color=C_ACTUAL, label="real")
    ax.plot(ga, [r["rms_sim"] for r in aq], "o-", color=C_MODEL, ms=6, label="quantile model")
    ax.plot(ga, [r["rms_sim"] for r in ag], "s--", color=C_BASE, ms=6, label="gaussian")
    ax.axvline(operating_gate, color=C_REF, ls=":", lw=1.5)
    _finish(ax, "T3a. RMS of errors that survive the gate\n"
                "this is what sets positioning error", "gate threshold (m)", "RMS (m)")

    ax = axes[0, 2]
    ax.axhline(0, color=C_REF, ls="--", lw=1.5, label="unbiased")
    ax.plot(ga, [r["bias_real"] for r in aq], "-", color=C_ACTUAL, label="real")
    ax.plot(ga, [r["bias_sim"] for r in aq], "o-", color=C_MODEL, ms=6, label="quantile model")
    ax.plot(ga, [r["bias_sim"] for r in ag], "s--", color=C_BASE, ms=6, label="gaussian")
    ax.axvline(operating_gate, color=C_REF, ls=":", lw=1.5)
    _finish(ax, "T3b. Bias of accepted errors\n"
                "skew leaves a mean the filter reads as real position offset",
            "gate threshold (m)", "mean of accepted (m)")

    ax = axes[1, 0]
    st = rep["stratum"]
    xs = np.arange(len(st["real"]))
    ax.fill_between(xs, st["lo"], st["hi"], color=C_ACTUAL, alpha=0.2, linewidth=0,
                    label="real 95% CI")
    ax.plot(xs, st["real"], "-", color=C_ACTUAL, label="real")
    ax.plot(xs, st["quantile"], "o-", color=C_MODEL, ms=6, label="quantile model")
    ax.plot(xs, st["gaussian"], "s--", color=C_BASE, ms=6, label="gaussian")
    _finish(ax, f"T4. Rejection rate by predicted-spread decile (g={operating_gate} m)\n"
                "marginal rate can match by accident; this cannot",
            "spread decile (narrow → wide)", "P(|e| > g)")

    ax = axes[1, 1]
    acc = lambda v: v[np.abs(v) <= operating_gate]
    qs = np.linspace(0.002, 0.998, 300)
    ref = np.quantile(acc(y), qs)
    for key, col, mk, nm in (("quantile", C_MODEL, "o", "quantile model"),
                             ("gaussian", C_BASE, "s", "gaussian")):
        src = rep["_S"][key].ravel()
        ax.plot(ref, np.quantile(acc(src), qs), mk, ms=3, color=col, label=nm)
    lim = [ref.min(), ref.max()]
    ax.plot(lim, lim, color=C_REF, ls="--", lw=1.5, label="y = x")
    _finish(ax, f"T4b. Q–Q of ACCEPTED errors (g={operating_gate} m)\n"
                "the distribution the filter actually ingests",
            "actual accepted quantile (m)", "simulated (m)")

    ax = axes[1, 2]
    pr = rep["proxy"]
    lo, hi = np.quantile(pr["real"][0], [0.002, 0.998])
    b = np.linspace(lo, hi, 90)
    # drop out-of-range rather than clipping it into a fake edge spike
    for key, col, nm in (("real", C_ACTUAL, "real"), ("quantile", C_MODEL, "quantile model"),
                         ("gaussian", C_BASE, "gaussian")):
        v = pr[key][0]
        ax.hist(v[(v >= lo) & (v <= hi)], bins=b, histtype="step", lw=2.0, density=True,
                color=col, label=f"{nm}  (outage {pr[key][1]:.2%})")
    _finish(ax, "T5. Position-error proxy: mean of gated errors\n"
                "end-to-end check — this is the quantity you care about",
            "proxy position error (m)", "density")

    fig.tight_layout()
    p = os.path.join(outdir, "03_gate.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# ======================================================================
# main
# ======================================================================
def main(X_train, y_train, X_val, y_val, features=None,
         groups_train=None, use_gpu=True, n_draws=200, outdir="qgbm_plots", seed=0,
         gate_thresholds=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0),
         operating_gate=3.0, n_anchors=4, refine_grid=False, support=None,
         tail_from=(0.05, 0.95), scale_from=None, calib_frac=0.25,
         plots=True, quiet=False):
    """Fit both models, score them distributionally, then score them the way
    an EKF gate study consumes them.

    gate_thresholds / operating_gate are in metres. Set operating_gate to the
    threshold your filter actually uses -- every T3/T4/T5 number is conditional
    on it, and the right model is the one that matches real data THERE, not on
    average across thresholds.

    refine_grid inserts quantile levels at each threshold's marginal quantile
    so the gate never lands mid-interpolation. Off by default -- on synthetic
    data it roughly doubled the model count for a negligible change in T2. It
    is an A/B to run once on real data, not a default to assume; see
    `taus_for_gate`.

    support passes a physical bound through to the sampler; None learns it from
    the training labels.
    """
    global TAUS
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    Xtr, Xva = _X(X_train, features), _X(X_val, features)
    ytr, yva = _y(y_train), _y(y_val)
    print(f"train {Xtr.shape}  val {Xva.shape}  gpu={use_gpu}")

    if refine_grid:
        TAUS = taus_for_gate(ytr, gate_thresholds)
        print(f"  grid refined around the gate thresholds: {len(TAUS)} levels\n"
              f"  {np.round(TAUS, 4)}")

    qm = QuantileGBM(taus=TAUS, use_gpu=use_gpu, seed=seed, support=support,
                     tail_from=tail_from, scale_from=scale_from) \
        .fit(Xtr, ytr, calib_frac=calib_frac, groups=groups_train)
    Q_raw = qm._grid(Xva)                          # raw grid: sampling + PIT
    Q_q = qm.predict_quantiles(Q=Q_raw)            # sampler's own quantiles: metrics
    print(f"\n  quantile crossing rate before sorting: {qm.crossing_rate_:.4%}"
          f"   (high => extreme levels underfit)")
    S_q = qm.sample(Q=Q_raw, n_draws=n_draws, rng=rng)
    pit_q = qm.cdf(yva, Q=Q_raw)

    gb = GaussianBaseline(use_gpu=use_gpu, seed=seed).fit(Xtr, ytr, groups=groups_train)
    Q_g = gb.predict_quantiles(Xva)
    S_g = gb.sample(Xva, n_draws=n_draws, rng=rng)
    pit_g = gb.cdf(yva, Xva)

    print("\n" + "=" * 78)
    print("DISTRIBUTIONAL METRICS  (correctness floor — not the acceptance criterion)")
    r_q = summarize("quantile GBM + empirical tails", yva, Q_q, S_q, pit_q)
    r_g = summarize("gaussian  mu + sigma*z  (current pipeline)", yva, Q_g, S_g, pit_g)
    d = (r_g["crps_samples"] - r_q["crps_samples"]) / r_g["crps_samples"]
    print(f"\n>> CRPS improvement of quantile model over gaussian: {d:+.2%}")

    i05, i95 = (int(np.argmin(np.abs(TAUS - t))) for t in (0.05, 0.95))
    spread = Q_q[:, i95] - Q_q[:, i05]
    rep = gate_report(yva, S_q, S_g, spread, gate_thresholds, operating_gate,
                      n_anchors=n_anchors, rng=rng,
                      handoff=(TAUS[qm.lo_i], TAUS[qm.hi_i]))
    rep["_S"] = {"quantile": S_q, "gaussian": S_g}

    if plots:
        p1 = plot_distributions(yva, S_q, Q_q, S_g, outdir)
        p2 = plot_calibration(yva, Q_q, pit_q, Q_g, pit_g, outdir)
        p3 = plot_gate(yva, rep, gate_thresholds, operating_gate, outdir)
        print(f"\nplots -> {p1}\n         {p2}\n         {p3}")

    return dict(model=qm, baseline=gb, Q=Q_q, samples=S_q, pit=pit_q,
                gate=rep, metrics={"quantile": r_q, "gaussian": r_g})


# ======================================================================
# A/B sweep over the tail configuration
# ======================================================================
def sweep(X_train, y_train, X_val, y_val, features=None, configs=None, **kw):
    """Run several tail configurations and print a one-line summary of each.

    The tail has two knobs that trade against each other -- where the handoff
    sits (how many exceedances the pool holds) and what the excesses are
    divided by (whether one pooled shape can serve every row). Which pair wins
    is an empirical question about YOUR error distribution, not something to
    reason out in advance. This runs the candidates and prints the numbers that
    decide it: pool dynamic range, PIT KS, and the rejection-rate ratio at the
    operating gate.

    Pick by the gate ratio first and PIT second -- they usually agree, and when
    they do not, the gate is what you consume.
    """
    if configs is None:
        configs = [
            dict(name="handoff .95 / scale .95", tail_from=(0.05, 0.95),
                 scale_from=(0.05, 0.95)),
            dict(name="handoff .95 / scale .99", tail_from=(0.05, 0.95),
                 scale_from=(0.01, 0.99)),
            dict(name="handoff .99 / scale .99", tail_from=(0.01, 0.99),
                 scale_from=(0.01, 0.99)),
        ]
    g = kw.get("operating_gate", 3.0)
    out = []
    for c in configs:
        nm = c.pop("name", str(c))
        print("\n" + "#" * 78 + f"\n### {nm}\n" + "#" * 78)
        r = main(X_train, y_train, X_val, y_val, features,
                 plots=False, **{**kw, **c})
        row = next(x for x in r["gate"]["rates"]["quantile"] if abs(x["g"] - g) < 1e-9)
        out.append(dict(name=nm, pit=r["metrics"]["quantile"]["pit_ks"],
                        crps=r["metrics"]["quantile"]["crps_samples"],
                        hi_dyn=(r["model"].hi_.pool[-1] /
                                max(np.median(r["model"].hi_.pool), 1e-9)),
                        gate_ratio=row["ratio"], gate_ok=row["ok"],
                        clamp=r["model"].clamp_rate_))
    print("\n" + "=" * 78)
    print(f"SWEEP SUMMARY   (gate ratio at g={g} m; 1.00 is perfect)")
    print(f"  {'config':<26} {'PIT KS':>8} {'CRPS':>9} {'pool dyn':>9} "
          f"{'gate x':>8} {'clamp':>8}")
    for o in out:
        print(f"  {o['name']:<26} {o['pit']:>8.4f} {o['crps']:>9.5f} "
              f"{o['hi_dyn']:>8.0f}x {o['gate_ratio']:>8.2f} {o['clamp']:>8.4%}"
              f"{'' if o['gate_ok'] else '   (gate MISS)'}")
    print("=" * 78)
    return out


from src.load import *
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # from your module:
    tags, labels, features = load_pos()
    range_error = labels["range_error_m"]
    DEFAULT_SPLIT = train_test_split(features, range_error, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = DEFAULT_SPLIT
    FEATURES = [f.value for f in PosFeature]
    #
    main(X_train, y_train, X_val, y_val, FEATURES,
         groups_train=None,   # pass your tag array here to keep grouping honest
         use_gpu=True,
         operating_gate=0.9)  # <-- SET THIS to your filter's actual threshold

    # To choose the tail configuration on your data rather than by argument:
    #   sweep(X_train, y_train, X_val, y_val, FEATURES,
    #         use_gpu=True, operating_gate=3.0)