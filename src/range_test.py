"""
Diagnostics for GaussianErrorModel, aimed at an EKF gate / positioning study.

    from gaussian_error_model import GaussianErrorModel
    from gaussian_error_tests import evaluate

    m = GaussianErrorModel().fit(X_train, y_train)
    evaluate(m, X_val, y_val, operating_gate=0.9)

Four figures, in the order you should read them:

  01_bias_model.png    is mu(x) right?     -- where the predicted mean lands
  02_sigma_model.png   is sigma(x) real?   -- and is the gaussian SHAPE ok
  03_distribution.png  does the sampler reproduce the noise?
  04_gate.png          does it behave correctly at your gate?

The split matters. A single "the model is calibrated" number cannot tell you
whether mu is biased, sigma is flat, or the shape is wrong, and those have
different fixes. The normalized residual z = (y - mu)/sigma separates them:
z off-centre means mu is wrong, sd(z) != 1 means sigma is wrong, z heavy-tailed
means the gaussian assumption is what is costing you.

Every comparison carries a reference for "how close is close enough" -- a
confidence interval, a critical value, or a control model -- because on 9k rows
almost any difference is statistically significant and almost none of them
matter.

The HOMOSCEDASTIC CONTROL (mu(x) + c*z, one constant c) answers the question
that comes after dropping the quantile model: is the variance model earning its
keep, or would a single number do? c is fitted to match 90% coverage on the
SAME data it is scored on, which is deliberately generous to the control -- if
sigma(x) still wins, that conclusion is safe.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; remove if you want interactive windows
import matplotlib.pyplot as plt

from src.range_model import _y

# ======================================================================
# palette / style (validated: lightness band, chroma, CVD separation,
# normal-vision separation, >=3:1 contrast on a #fcfcfb surface)
# ======================================================================
C_ACTUAL = "#eb6834"   # orange - observed / real
C_MODEL = "#2a78d6"    # blue   - the model
C_BASE = "#12946a"     # green  - homoscedastic control
C_REF = "#898781"      # muted  - reference / nominal
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"

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

ROBUST = 1.3489795  # IQR of a standard normal; IQR/this is a fat-tail-resistant sd


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


def _bins(v, nbins):
    """Equal-count bins. Quantile edges, not equal-width: with a skewed
    predictor, equal-width bins put 95% of rows in the first one."""
    e = np.quantile(v, np.linspace(0, 1, nbins + 1))
    e[-1] += 1e-9
    return np.clip(np.digitize(v, e[1:-1]), 0, nbins - 1)


def _wilson(k, n, z=1.96):
    """Wilson score interval.

    The textbook p +- z*sqrt(p(1-p)/n) collapses to ZERO WIDTH at p=0, which is
    common in the narrow-sigma strata where nothing exceeds the gate. A
    zero-width interval flags every simulated rate as a miss, reporting
    failures that are artifacts of the arithmetic. Wilson gives [0, ~z^2/n]
    there instead.
    """
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(c - h, 0.0)), float(min(c + h, 1.0))


# ======================================================================
# 01 -- is mu(x) right?
# ======================================================================
def plot_bias_model(y, mu, sigma, outdir, nbins=20):
    y = _y(y)
    r = y - mu
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. reliability: does the predicted mean land where it says?
    ax = axes[0, 0]
    idx = _bins(mu, nbins)
    px, ry, lo, hi = [], [], [], []
    for b in range(nbins):
        m = idx == b
        if not m.sum():
            continue
        se = y[m].std() / np.sqrt(m.sum())
        px.append(mu[m].mean())
        ry.append(y[m].mean())
        lo.append(y[m].mean() - 1.96 * se)
        hi.append(y[m].mean() + 1.96 * se)
    lim = [min(min(px), min(ry)), max(max(px), max(ry))]
    ax.plot(lim, lim, color=C_REF, ls="--", lw=1.5, label="y = x (calibrated)")
    ax.vlines(px, lo, hi, color=C_MODEL, lw=1.2, alpha=0.6)
    ax.plot(px, ry, "o", ms=6, color=C_MODEL, label="realized mean")
    _finish(ax, "1. Bias reliability — predicted mean vs realized\n"
                "on the diagonal = mu(x) means what it says",
            "predicted mu (m)", "realized mean of y (m)")

    # 2/3. residual mean by mu-stratum and by sigma-stratum
    for ax, v, nm, num in ((axes[0, 1], mu, "predicted mu", 2),
                           (axes[1, 0], sigma, "predicted sigma", 3)):
        i2 = _bins(v, 10)
        mm, ll, hh = [], [], []
        for b in range(10):
            m = i2 == b
            se = r[m].std() / np.sqrt(max(m.sum(), 1))
            mm.append(r[m].mean())
            ll.append(r[m].mean() - 1.96 * se)
            hh.append(r[m].mean() + 1.96 * se)
        ax.axhline(0, color=C_REF, ls="--", lw=1.5, label="unbiased")
        ax.fill_between(range(10), ll, hh, color=C_MODEL, alpha=0.2, linewidth=0,
                        label="95% CI")
        ax.plot(range(10), mm, "o-", ms=6, color=C_MODEL, label="mean residual")
        _finish(ax, f"{num}. Residual mean by {nm} decile\n"
                    f"flat on zero = no leftover structure on this axis",
                f"{nm} decile (low → high)", "mean of y − mu (m)")

    # 4. how much did the mean model actually explain
    ax = axes[1, 1]
    lo_, hi_ = np.quantile(y, [0.002, 0.998])
    b = np.linspace(lo_, hi_, 90)
    ax.hist(y[(y >= lo_) & (y <= hi_)], bins=b, color=C_ACTUAL, alpha=0.6,
            density=True, label=f"y  (sd {y.std():.3f})")
    ax.hist(r[(r >= lo_) & (r <= hi_)], bins=b, color=C_MODEL, alpha=0.6,
            density=True, label=f"y − mu  (sd {r.std():.3f})")
    r2 = 1 - r.var() / y.var()
    _finish(ax, f"4. Variance explained by mu(x):  R² = {r2:.3f}\n"
                "the blue mass is what sigma(x) must describe",
            "range error (m)", "density")

    fig.tight_layout()
    p = os.path.join(outdir, "01_bias_model.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# ======================================================================
# 02 -- is sigma(x) real, and is the gaussian shape ok?
# ======================================================================
def plot_sigma_model(y, mu, sigma, outdir, nbins=10):
    from scipy.stats import norm
    y = _y(y)
    r = y - mu
    z = r / sigma
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    idx = _bins(sigma, nbins)

    # 1. predicted spread vs realized spread
    ax = axes[0, 0]
    pr, sd, rb = [], [], []
    for b in range(nbins):
        m = idx == b
        q = np.percentile(r[m], [25, 75])
        pr.append(sigma[m].mean())
        sd.append(r[m].std())
        rb.append((q[1] - q[0]) / ROBUST)
    lim = [0, max(max(pr), max(sd)) * 1.05]
    ax.plot(lim, lim, color=C_REF, ls="--", lw=1.5, label="y = x (calibrated)")
    ax.plot(pr, sd, "o-", ms=6, color=C_MODEL, label="realized sd")
    ax.plot(pr, rb, "s--", ms=6, color=C_BASE, label="realized IQR/1.349 (robust)")
    _finish(ax, "1. Predicted vs realized spread, by sigma decile\n"
                "gap between the two series = how fat the tails are",
            "predicted sigma (m)", "realized spread of y − mu (m)")

    # 2. the cleanest single test of sigma
    ax = axes[0, 1]
    zs = [z[idx == b].std() for b in range(nbins)]
    zr = [(lambda q: (q[1] - q[0]) / ROBUST)(np.percentile(z[idx == b], [25, 75]))
          for b in range(nbins)]
    ax.axhline(1.0, color=C_REF, ls="--", lw=1.5, label="calibrated")
    ax.plot(range(nbins), zs, "o-", ms=6, color=C_MODEL, label="sd(z)")
    ax.plot(range(nbins), zr, "s--", ms=6, color=C_BASE, label="robust sd(z)")
    ax.set_ylim(0, max(2.0, max(zs) * 1.15))
    _finish(ax, "2. Normalized residual spread by sigma decile\n"
                "flat on 1.0 = sigma(x) tracks; sloped = it is an average",
            "sigma decile (narrow → wide)", "sd of (y − mu)/sigma")

    # 3. how much heteroscedasticity is there to learn
    ax = axes[1, 0]
    ax.hist(sigma, bins=np.geomspace(sigma.min(), sigma.max(), 70),
            color=C_MODEL, alpha=0.8)
    ax.set_xscale("log")
    ax.axvline(np.median(sigma), color=C_REF, ls="--", lw=1.5,
               label=f"median {np.median(sigma):.3f} m")
    _finish(ax, f"3. Distribution of sigma(x)   "
                f"({sigma.max() / np.median(sigma):.0f}x median at the top)\n"
                "a spike = the model found no heteroscedasticity",
            "predicted sigma (m), log scale", "rows")

    # 4. the gaussian assumption itself
    ax = axes[1, 1]
    n = len(z)
    take = np.unique(np.linspace(0, n - 1, min(n, 3000)).astype(int))
    theo = norm.ppf((np.arange(1, n + 1) - 0.5) / n)[take]
    ax.plot(theo, np.sort(z)[take], "o", ms=3, color=C_MODEL, label="z = (y − mu)/sigma")
    lim = [theo.min() * 1.05, theo.max() * 1.05]
    ax.plot(lim, lim, color=C_REF, ls="--", lw=1.5, label="N(0,1)")
    _finish(ax, "4. Q–Q of the normalized residual vs N(0,1)\n"
                "ends above/below the line = the fat tails this model cannot represent",
            "theoretical normal quantile", "observed z")

    fig.tight_layout()
    p = os.path.join(outdir, "02_sigma_model.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# ======================================================================
# 03 -- does the sampler reproduce the noise?
# ======================================================================
def plot_distribution(y, S, S_h, pit, outdir, ctrl="homoscedastic"):
    y = _y(y)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    lo, hi = np.quantile(y, [0.002, 0.995])
    b = np.linspace(lo, hi, 120)
    for v, c, nm in ((y, C_ACTUAL, "actual"), (S[:, 0], C_MODEL, "sampled")):
        ax.hist(v[(v >= lo) & (v <= hi)], bins=b, color=c, alpha=0.6, density=True,
                label=f"{nm}  ({np.mean((v < lo) | (v > hi)):.2%} off-axis)")
    _finish(ax, "1. Marginal — one synthetic replica vs actual\n"
                "same n, same feature mix; these SHOULD overlap",
            "range error (m)", "density")

    ax = axes[0, 1]
    n = len(pit)
    ks = float(np.max(np.abs(np.sort(pit) - np.linspace(0, 1, n))))
    crit = 1.36 / np.sqrt(n)
    ax.hist(pit, bins=25, color=C_MODEL, alpha=0.8, density=True,
            label=f"KS {ks:.4f} (5% crit {crit:.4f})")
    ax.axhline(1.0, color=C_REF, ls="--", lw=1.5, label="uniform (calibrated)")
    ax.set_ylim(0, max(2.5, np.histogram(pit, 25, density=True)[0].max() * 1.15))
    _finish(ax, "2. PIT histogram\n"
                "U-shape = too narrow, hump = too wide, tilt = biased",
            "PIT value", "density")

    ax = axes[1, 0]
    ts = np.linspace(0, np.quantile(np.abs(y), 0.9999), 200)
    surv = lambda v: [(np.abs(v) > t).mean() for t in ts]
    ax.semilogy(ts, surv(y), color=C_ACTUAL, label="actual")
    ax.semilogy(ts, surv(S.ravel()), color=C_MODEL, label="model")
    ax.semilogy(ts, surv(S_h.ravel()), color=C_BASE, ls="--", label=ctrl)
    ax.set_ylim(1e-5, 1.5)
    _finish(ax, "3. Gate curve — P(|e| > t), log scale\n"
                "this IS the rejection-rate curve at every possible threshold",
            "gate threshold t (m)", "P(|e| > t)")

    ax = axes[1, 1]
    q = np.linspace(0.001, 0.999, 400)
    ax.plot(np.quantile(y, q), np.quantile(S.ravel(), q), "o", ms=3,
            color=C_MODEL, label="model")
    ax.plot(np.quantile(y, q), np.quantile(S_h.ravel(), q), "s", ms=3,
            color=C_BASE, label=ctrl)
    lim = [np.quantile(y, 0.001), np.quantile(y, 0.999)]
    ax.plot(lim, lim, color=C_REF, ls="--", lw=1.5, label="y = x")
    _finish(ax, "4. Q–Q, simulated vs actual\n"
                "departures at the ends are what the gate sees",
            "actual quantile (m)", "simulated quantile (m)")

    fig.tight_layout()
    p = os.path.join(outdir, "03_distribution.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# ======================================================================
# gate tests
# ======================================================================
def _replicas(S, n_rep=20):
    """Each column of S is a synthetic dataset of the same size and feature mix
    as the real val set -- the apples-to-apples object for a two-sample test."""
    m = S.shape[1]
    return [S[:, i] for i in np.unique(np.linspace(0, m - 1, min(n_rep, m)).astype(int))]


def marginal_fidelity(y, S, n_rep=20):
    from scipy.stats import ks_2samp, wasserstein_distance
    y = _y(y)
    reps = _replicas(S, n_rep)
    ks = np.array([ks_2samp(y, r).statistic for r in reps])
    w = np.array([wasserstein_distance(y, r) for r in reps])
    crit = 1.36 * np.sqrt(2.0 / len(y))
    return dict(ks=float(ks.mean()), ks_sd=float(ks.std()), crit=float(crit),
                ok=bool(ks.mean() < crit), w=float(w.mean()),
                sd_real=float(y.std()), sd_sim=float(np.mean([r.std() for r in reps])))


def gate_rates(y, S, thresholds):
    y = _y(y)
    n, pool = len(y), S.ravel()
    out = []
    for g in thresholds:
        k = int((np.abs(y) > g).sum())
        p = k / n
        lo, hi = _wilson(k, n)
        ps = float(np.mean(np.abs(pool) > g))
        out.append(dict(g=float(g), real=p, lo=lo, hi=hi, sim=ps, n_real=k,
                        ratio=float(ps / p) if p > 0 else np.nan,
                        ok=bool(lo <= ps <= hi)))
    return out


def accepted_moments(y, S, thresholds, n_rep=20):
    """Moments of the errors that SURVIVE the gate -- these, not the rejected
    blunders, are what corrupt the position solution. `bias` matters most: a
    non-zero mean among accepted measurements is indistinguishable to the
    filter from a real position offset."""
    y = _y(y)
    reps = _replicas(S, n_rep)
    out = []
    for g in thresholds:
        a = y[np.abs(y) <= g]
        sim = [r[np.abs(r) <= g] for r in reps]
        sim = [s for s in sim if len(s) > 10]
        if len(a) < 10 or not sim:
            continue
        out.append(dict(g=float(g),
                        rms_real=float(np.sqrt(np.mean(a ** 2))),
                        rms_sim=float(np.mean([np.sqrt(np.mean(s ** 2)) for s in sim])),
                        bias_real=float(a.mean()),
                        bias_sim=float(np.mean([s.mean() for s in sim]))))
    return out


def gate_by_stratum(y, S, strat, g, nbins=10, n_rep=20):
    """Rejection rate within strata of predicted sigma.

    The marginal rate can match by accident -- too many rejections in the easy
    rows cancelling too few in the hard ones. This is the conditional version.
    It matters for a gate study because rejections are not spread evenly: if
    the model puts them in the wrong rows, the simulated outage PATTERN is
    wrong even when the simulated outage RATE is right."""
    y = _y(y)
    idx = _bins(strat, nbins)
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
        sim.append(float(np.mean([np.mean(np.abs(r[m]) > g) for r in reps])) if nb
                   else np.nan)
    return np.array(real), np.array(sim), np.array(lo), np.array(hi)


def position_proxy(pool, g, n_anchors=4, n_trials=200_000, rng=None):
    """First-order stand-in for the position error the filter experiences:
    draw n_anchors i.i.d. errors, gate them, average the survivors. Under
    isotropic geometry with DOP=1 the averaged residual is proportional to the
    position error. Returns (proxy errors, outage rate); outage = every anchor
    rejected, a failure mode the gate threshold directly controls and which no
    marginal comparison can see."""
    rng = rng or np.random.default_rng(0)
    pool = np.asarray(pool, float)
    E = pool[rng.integers(0, len(pool), size=(n_trials, n_anchors))]
    keep = np.abs(E) <= g
    nk = keep.sum(axis=1)
    ok = nk > 0
    est = np.divide((E * keep).sum(axis=1), np.maximum(nk, 1))
    return est[ok], float(np.mean(~ok))


def plot_gate(y, S, S_h, sigma, thresholds, g0, rep, outdir, n_anchors=4, rng=None,
              ctrl="homoscedastic"):
    y = _y(y)
    rng = rng or np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    gs = np.array([r["g"] for r in rep["rates"]])

    ax = axes[0, 0]
    ax.fill_between(gs, [r["lo"] for r in rep["rates"]], [r["hi"] for r in rep["rates"]],
                    color=C_ACTUAL, alpha=0.2, linewidth=0, label="real 95% CI")
    ax.semilogy(gs, [r["real"] for r in rep["rates"]], "-", color=C_ACTUAL, label="real")
    ax.semilogy(gs, [r["sim"] for r in rep["rates"]], "o-", ms=6, color=C_MODEL,
                label="model")
    ax.semilogy(gs, [r["sim"] for r in rep["rates_h"]], "s--", ms=6, color=C_BASE,
                label=ctrl)
    ax.axvline(g0, color=C_REF, ls=":", lw=1.5)
    _finish(ax, "T2. Gate rejection rate vs threshold\n"
                "inside the band = the gate fires as often as in reality",
            "gate threshold (m)", "P(|e| > g)")

    ga = np.array([r["g"] for r in rep["accepted"]])
    for ax, key, ttl, sub, yl in (
            (axes[0, 1], "rms", "T3a. RMS of errors that survive the gate",
             "this is what sets positioning error", "RMS (m)"),
            (axes[0, 2], "bias", "T3b. Bias of accepted errors",
             "a mean the filter reads as a real position offset", "mean of accepted (m)")):
        if key == "bias":
            ax.axhline(0, color=C_REF, ls="--", lw=1.5, label="unbiased")
        ax.plot(ga, [r[f"{key}_real"] for r in rep["accepted"]], "-",
                color=C_ACTUAL, label="real")
        ax.plot(ga, [r[f"{key}_sim"] for r in rep["accepted"]], "o-", ms=6,
                color=C_MODEL, label="model")
        ax.plot(ga, [r[f"{key}_sim"] for r in rep["accepted_h"]], "s--", ms=6,
                color=C_BASE, label=ctrl)
        ax.axvline(g0, color=C_REF, ls=":", lw=1.5)
        _finish(ax, f"{ttl}\n{sub}", "gate threshold (m)", yl)

    ax = axes[1, 0]
    st = rep["stratum"]
    xs = np.arange(len(st["real"]))
    ax.fill_between(xs, st["lo"], st["hi"], color=C_ACTUAL, alpha=0.2, linewidth=0,
                    label="real 95% CI")
    ax.plot(xs, st["real"], "-", color=C_ACTUAL, label="real")
    ax.plot(xs, st["model"], "o-", ms=6, color=C_MODEL, label="model")
    ax.plot(xs, st["homo"], "s--", ms=6, color=C_BASE, label=ctrl)
    _finish(ax, f"T4. Rejection rate by predicted-sigma decile (g={g0} m)\n"
                "marginal rate can match by accident; this cannot",
            "sigma decile (narrow → wide)", "P(|e| > g)")

    ax = axes[1, 1]
    acc = lambda v: v[np.abs(v) <= g0]
    qs = np.linspace(0.002, 0.998, 300)
    ref = np.quantile(acc(y), qs)
    ax.plot(ref, np.quantile(acc(S.ravel()), qs), "o", ms=3, color=C_MODEL, label="model")
    ax.plot(ref, np.quantile(acc(S_h.ravel()), qs), "s", ms=3, color=C_BASE,
            label=ctrl)
    ax.plot([ref.min(), ref.max()], [ref.min(), ref.max()], color=C_REF, ls="--",
            lw=1.5, label="y = x")
    _finish(ax, f"T4b. Q–Q of ACCEPTED errors (g={g0} m)\n"
                "the distribution the filter actually ingests",
            "actual accepted quantile (m)", "simulated (m)")

    ax = axes[1, 2]
    pr = rep["proxy"]
    lo, hi = np.quantile(pr["real"][0], [0.002, 0.998])
    b = np.linspace(lo, hi, 90)
    for key, col, nm in (("real", C_ACTUAL, "real"), ("model", C_MODEL, "model"),
                         ("homo", C_BASE, ctrl)):
        v = pr[key][0]
        ax.hist(v[(v >= lo) & (v <= hi)], bins=b, histtype="step", lw=2.0,
                density=True, color=col, label=f"{nm}  (outage {pr[key][1]:.2%})")
    _finish(ax, "T5. Position-error proxy: mean of gated errors\n"
                "end-to-end check — this is the quantity you care about",
            "proxy position error (m)", "density")

    fig.tight_layout()
    p = os.path.join(outdir, "04_gate.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


# ======================================================================
# entry point
# ======================================================================
def evaluate(model, X_val, y_val, operating_gate=0.9,
             gate_thresholds=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0),
             n_draws=200, n_anchors=4, outdir="gauss_plots", seed=0, n_rep=20,
             control="auto"):
    """Full diagnostic run. Returns every number computed; writes 4 figures.

    operating_gate is in metres and must be YOUR filter's threshold: T3, T4 and
    T5 are all conditional on it, and the model is 'good' or 'bad' relative to
    that point, not on average across the sweep.

    control picks the green comparison series:
      "homoscedastic"  mu(x) + c*z with one constant c  -- is sigma(x) worth it?
      "gaussian"       the same model with the gaussian sampler -- is the
                       empirical shape worth it?
      "auto"           "gaussian" when the model uses sampler="empirical",
                       otherwise "homoscedastic"
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    y = _y(y_val)
    gate_thresholds = tuple(sorted(set(list(gate_thresholds) + [operating_gate])))
    if control == "auto":
        control = "gaussian" if model.sampler == "empirical" else "homoscedastic"

    mu, sigma = model.predict_mu_sigma(X_val)
    S = model.sample(X_val, n_draws=n_draws, rng=rng)
    pit = model.cdf(y, X_val)
    z = (y - mu) / sigma

    c = float(np.quantile(np.abs(y - mu), 0.90) / 1.6449)
    if control == "gaussian":
        ctrl = "gaussian sampler"
        S_h = model.sample(X_val, n_draws=n_draws, rng=rng, sampler="gaussian")
    elif control == "homoscedastic":
        # deliberately fitted on the data it is scored on -- generous to the control
        ctrl = "homoscedastic"
        S_h = model._clamp(mu[:, None] + c * rng.standard_normal((len(mu), n_draws)))
    else:
        raise ValueError(f"unknown control {control!r}")
    cs_ = "gauss" if control == "gaussian" else "homo"

    n = len(y)
    ks = float(np.max(np.abs(np.sort(pit) - np.linspace(0, 1, n))))
    crit = 1.36 / np.sqrt(n)
    cov = float(np.mean(np.abs(z) <= 1.6449))

    print("\n" + "=" * 78)
    print(f"GAUSSIAN ERROR MODEL — DIAGNOSTICS   (val n={n}, {n_draws} draws/row)")
    print(f"   sampler = {model.sampler}   control = {ctrl}")
    print("=" * 78)
    print(f"\n-- model")
    print(f"   k (sigma recal)      {model.k_:.4f}")
    print(f"   R^2 of mu(x)         {1 - (y - mu).var() / y.var():.4f}")
    print(f"   sigma  median        {np.median(sigma):.4f} m   "
          f"range [{sigma.min():.4f}, {sigma.max():.4f}]")
    if control == "homoscedastic":
        print(f"   homoscedastic c      {c:.4f} m   (the control's single sigma)")
    print(f"\n-- normalized residual  z = (y - mu)/sigma   [target: N(0,1)]")
    print(f"   mean   {z.mean():+.4f}   (0 => mu unbiased)")
    print(f"   sd     {z.std():.4f}   (1 => sigma right on average)")
    print(f"   robust sd {(np.percentile(z, 75) - np.percentile(z, 25)) / ROBUST:.4f}"
          f"   kurtosis {float(((z - z.mean()) ** 4).mean() / z.var() ** 2):.2f}"
          f"   (3 => gaussian; higher => fat tails)")
    print(f"\n-- calibration")
    print(f"   PIT KS  {ks:.4f}   (5% crit {crit:.4f} -> "
          f"{'CALIBRATED' if ks < crit else 'REJECT uniformity'})")
    print(f"   90% coverage  {cov:.3%}   (nominal 90%)")
    print(f"   sample range  [{S.min():.3g}, {S.max():.3g}]   "
          f"(observed [{y.min():.3g}, {y.max():.3g}])")

    mf, mf_h = marginal_fidelity(y, S, n_rep), marginal_fidelity(y, S_h, n_rep)
    print(f"\n-- T1. Marginal fidelity   (two-sample KS 5% crit {mf['crit']:.4f})")
    for nm, d in (("model", mf), (cs_, mf_h)):
        print(f"   {nm:<11} KS={d['ks']:.4f} (+-{d['ks_sd']:.4f})  W1={d['w']:.4f}  "
              f"sd={d['sd_sim']:.4f} vs real {d['sd_real']:.4f}"
              f"   {'PASS' if d['ok'] else 'FAIL'}")

    gr, gr_h = gate_rates(y, S, gate_thresholds), gate_rates(y, S_h, gate_thresholds)
    print(f"\n-- T2. Gate rejection rate P(|e| > g)")
    print(f"   {'gate(m)':>8} {'real':>9} {'95% CI (Wilson)':>19} {'n_evt':>6} "
          f"{'model':>9} {'x':>6} {'':>5} {cs_:>9} {'x':>6}")
    for a, b in zip(gr, gr_h):
        star = " <" if abs(a["g"] - operating_gate) < 1e-9 else ""
        print(f"   {a['g']:>8.2f} {a['real']:>9.5f} [{a['lo']:.5f},{a['hi']:.5f}] "
              f"{a['n_real']:>6d} {a['sim']:>9.5f} {a['ratio']:>6.2f} "
              f"{'ok' if a['ok'] else 'MISS':>5} {b['sim']:>9.5f} {b['ratio']:>6.2f} "
              f"{'ok' if b['ok'] else 'MISS'}{star}")
    print("   (x = sim/real. Read the ratio, not just ok/MISS: at large n_evt a 5% "
          "error is a MISS.)")

    am, am_h = (accepted_moments(y, S, gate_thresholds, n_rep),
                accepted_moments(y, S_h, gate_thresholds, n_rep))
    print(f"\n-- T3. Errors that SURVIVE the gate")
    print(f"   {'gate(m)':>8} {'RMS real':>9} {'RMS mdl':>9} {'RMS ' + cs_:>9} "
          f"{'bias real':>10} {'bias mdl':>10} {'bias ' + cs_:>10}")
    for a, b in zip(am, am_h):
        print(f"   {a['g']:>8.2f} {a['rms_real']:>9.4f} {a['rms_sim']:>9.4f} "
              f"{b['rms_sim']:>9.4f} {a['bias_real']:>10.4f} {a['bias_sim']:>10.4f} "
              f"{b['bias_sim']:>10.4f}")

    cr, cs, clo, chi = gate_by_stratum(y, S, sigma, operating_gate, n_rep=n_rep)
    _, ch, _, _ = gate_by_stratum(y, S_h, sigma, operating_gate, n_rep=n_rep)
    print(f"\n-- T4. Conditional rejection rate by sigma decile (g={operating_gate} m)")
    print(f"   deciles outside the real 95% CI:  model "
          f"{int(np.sum((cs < clo) | (cs > chi)))}/10   {ctrl} "
          f"{int(np.sum((ch < clo) | (ch > chi)))}/10   (0-1 is healthy)")

    pr = {k: position_proxy(v, operating_gate, n_anchors, rng=rng)
          for k, v in (("real", y), ("model", S.ravel()), ("homo", S_h.ravel()))}
    print(f"\n-- T5. Position-error proxy over {n_anchors} anchors "
          f"(g={operating_gate} m)")
    print(f"   {'source':<14} {'RMS':>9} {'bias':>9} {'P95|e|':>9} {'outage':>9}")
    for k, nm in (("real", "real"), ("model", "model"), ("homo", ctrl)):
        v = pr[k][0]
        print(f"   {nm:<16} {np.sqrt(np.mean(v ** 2)):>9.4f} {v.mean():>9.4f} "
              f"{np.percentile(np.abs(v), 95):>9.4f} {pr[k][1]:>9.5f}")
    print("=" * 78)

    rep = dict(rates=gr, rates_h=gr_h, accepted=am, accepted_h=am_h,
               stratum=dict(real=cr, model=cs, homo=ch, lo=clo, hi=chi), proxy=pr)

    p1 = plot_bias_model(y, mu, sigma, outdir)
    p2 = plot_sigma_model(y, mu, sigma, outdir)
    p3 = plot_distribution(y, S, S_h, pit, outdir, ctrl=ctrl)
    p4 = plot_gate(y, S, S_h, sigma, gate_thresholds, operating_gate, rep, outdir,
                   n_anchors=n_anchors, rng=rng, ctrl=ctrl)
    print(f"\nplots -> {p1}\n         {p2}\n         {p3}\n         {p4}")

    return dict(mu=mu, sigma=sigma, z=z, pit=pit, samples=S, samples_homo=S_h,
                pit_ks=ks, pit_crit=crit, cov90=cov, homo_c=c,
                marginal=dict(model=mf, homo=mf_h), gate=rep)


if __name__ == "__main__":
    from src.load import *
    from sklearn.model_selection import train_test_split
    from src.range_model import GaussianErrorModel

    tags, labels, features = load_pos()
    range_error = labels["range_error_m"]
    X_train, X_val, y_train, y_val = train_test_split(
        features, range_error, test_size=0.2, random_state=42)
    FEATURES = [f.value for f in PosFeature]

    model = GaussianErrorModel(use_gpu=True,
                               sampler="empirical",   # "gaussian" for the old behaviour
                               n_strata=5,            # try 3-5 if T4 misses in wide rows
                               ).fit(X_train, y_train, features=FEATURES)
    evaluate(model, X_val, y_val,
             operating_gate=0.9)   # <-- your filter's actual threshold
    # green series is the SAME model with the gaussian sampler, so every plot
    # is a direct before/after of the shape change.
    # evaluate(model, X_val, y_val, operating_gate=0.9, control="homoscedastic")