from scipy.stats import wasserstein_distance
import numpy as np
import polars as pl
import plotly.graph_objects as go

# Shared metrics for the measurement error tests (range_tests, angle_tests)
# Every model is represented by a draws matrix of shape (n_points, n_draws) : n_draws independent samples of the error
# for each evaluation point, so baselines just need to output the same format


# Scene = (scenario_id, drop_id), same grouping as group_split
def scene_ids(tags : pl.DataFrame):
    return tags.select(pl.struct(["scenario_id", "drop_id"]).hash()).to_series().to_numpy()


# Repeat every row n times, block order : row r of copy k lands at k * len(X) + r
def tile_frame(X : pl.DataFrame, n : int):
    return pl.concat([X] * n, how = "vertical")

# Inverse of tile_frame for a flat array of samples -> (n_points, n_draws)
def to_draws(flat, n_points : int, n_draws : int):
    return np.asarray(flat).reshape(n_draws, n_points).T


# (1) W1 (1D earth mover's distance) between the real errors and the model's pooled marginal

def w1(real, draws):
    return wasserstein_distance(np.asarray(real), np.asarray(draws).ravel())


# (2) Monte Carlo null test for W1
# Assume the model is the true data generating process : then the real errors are just one more draw, exchangeable
# with the K synthetic draws. Pool Z = [y, D_1, ..., D_K], score each member by its mean W1 to all other members,
# and take the rank of y. Any nonzero W1 between synthetic draws is purely finite sample noise
# p value = fraction of synthetic scores at least as large as the real one (+1 so p is never exactly 0)

def w1_null(real, draws):
    Z = np.column_stack([np.asarray(real), np.asarray(draws)])
    Z = np.sort(Z, axis = 0)
    k = Z.shape[1]

    # equal sample sizes, so W1 = mean |sorted a - sorted b|
    dist = np.zeros((k, k))
    for i in range(k):
        dist[i] = np.abs(Z[:, [i]] - Z).mean(axis = 0)

    scores = dist.sum(axis = 1) / (k - 1)
    observed, null = scores[0], scores[1:]
    p_value = (1 + np.sum(null >= observed)) / (1 + len(null))
    return observed, null, p_value


def plot_w1_null(observed, null, p_value, title : str, unit : str):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x = null, nbinsx = 30, name = "W1 under null", opacity = 0.7))
    fig.add_vline(x = observed, line = dict(color = "firebrick", width = 2, dash = "dash"),
                  annotation_text = f"ISAC errors: {observed:.4f}", annotation_position = "top")
    fig.update_layout(
        title = f"MC Null W1 {title} ({len(null)} sims) : p = {p_value:.4f}",
        xaxis_title = f"mean W1 to other draws ({unit})",
        yaxis_title = "count",
        template = "simple_white"
    )
    fig.show()


# (3) CRPS, fair ensemble estimator per point
# CRPS = E|X - y| - 1/2 E|X - X'|, with the pair term over the m(m - 1) distinct pairs so it's unbiased for finite m
# sum_{j,k} |x_j - x_k| = 2 * sum_i (2i - m - 1) x_(i) on the sorted draws

def crps_ensemble(real, draws):
    real = np.asarray(real)[:, None]
    draws = np.asarray(draws)
    m = draws.shape[1]

    abs_err = np.abs(draws - real).mean(axis = 1)
    if m == 1:
        return abs_err

    x = np.sort(draws, axis = 1)
    i = np.arange(1, m + 1)
    pair_sum = 2 * ((2 * i - m - 1) * x).sum(axis = 1)
    return abs_err - pair_sum / (2 * m * (m - 1))


# (4) Bootstrap by scene : resample whole scenes with replacement, since points within a scene are correlated
# stat_fn(idx) -> array of statistics, so related statistics (eg CRPS and CRPSS) share the same resamples

def scene_bootstrap(stat_fn, scene, n_boot : int = 1000, alpha : float = 0.05, seed : int = 42):
    rng = np.random.default_rng(seed = seed)
    scenes = np.unique(scene)
    rows_by_scene = {s : np.flatnonzero(scene == s) for s in scenes}

    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(scenes, size = len(scenes), replace = True)
        idx = np.concatenate([rows_by_scene[s] for s in chosen])
        boots.append(np.atleast_1d(stat_fn(idx)))

    boots = np.array(boots)
    return np.nanquantile(boots, alpha / 2, axis = 0), np.nanquantile(boots, 1 - alpha / 2, axis = 0)


# (5) Score every model : pooled W1 + MC null p, and optionally CRPS / CRPSS against a reference model
# models : {name : draws}

def score_models(real, scene, models : dict, crps : bool = False, ref : str = None,
                 n_boot : int = 1000, seed : int = 42):
    real = np.asarray(real)
    rows = []
    if crps and ref is not None:
        ref_crps = crps_ensemble(real, models[ref])

    for name, draws in models.items():
        draws = np.asarray(draws)
        _, _, p_value = w1_null(real, draws)
        point = {"w1" : w1(real, draws)}

        if crps:
            crps_rows = crps_ensemble(real, draws)
            point["crps"] = crps_rows.mean()
            if ref is not None:
                point["crpss"] = 1 - crps_rows.mean() / ref_crps.mean()

        def stat_fn(idx):
            out = [w1(real[idx], draws[idx])]
            if crps:
                out.append(crps_rows[idx].mean())
                if ref is not None:
                    out.append(1 - crps_rows[idx].mean() / ref_crps[idx].mean())
            return np.array(out)

        lo, hi = scene_bootstrap(stat_fn, scene, n_boot, seed = seed)
        for (metric, val), l, h in zip(point.items(), lo, hi):
            rows.append({"model" : name, "metric" : metric, "value" : val, "lo" : l, "hi" : h})
        rows.append({"model" : name, "metric" : "w1_null_p", "value" : p_value, "lo" : None, "hi" : None})

    return pl.DataFrame(rows)


# Wide table : rows = models, columns = metrics, cells = "value [lo, hi]"
def print_table(scores : pl.DataFrame, title : str = ""):
    scores = scores.with_columns(
        pl.when(pl.col("lo").is_null())
          .then(pl.col("value").round(4).cast(pl.String))
          .otherwise(pl.format("{} [{}, {}]", pl.col("value").round(4), pl.col("lo").round(4), pl.col("hi").round(4)))
          .alias("cell")
    )
    print(f"\n{title}")
    print(scores.pivot(on = "metric", index = "model", values = "cell"))


# (6) W1 (and optionally CRPS) per equal mass bin of a covariate
# Pooled W1 only checks the marginal, so an unconditional baseline gets it nearly right by construction.
# Slicing checks the conditional p(error | feature)

def slice_metrics(X : pl.DataFrame, real, models : dict, feature : str, n_bins : int = 8, crps : bool = False):
    real = np.asarray(real)
    col = X[feature].cast(pl.Float64).to_numpy()
    valid = ~np.isnan(col)
    order = np.argsort(col[valid])
    chunks = np.array_split(np.flatnonzero(valid)[order], n_bins)

    rows = []
    for name, draws in models.items():
        draws = np.asarray(draws)
        for idx in chunks:
            row = {
                "model" : name,
                "f_mid" : col[idx].mean(),
                "n" : len(idx),
                "w1" : w1(real[idx], draws[idx])
            }
            if crps:
                row["crps"] = crps_ensemble(real[idx], draws[idx]).mean()
            rows.append(row)
    return pl.DataFrame(rows)


def plot_slices(slices : pl.DataFrame, feature : str, title : str = ""):
    for metric in [m for m in ["w1", "crps"] if m in slices.columns]:
        fig = go.Figure()
        for name in slices["model"].unique(maintain_order = True):
            s = slices.filter(pl.col("model") == name)
            fig.add_trace(go.Scatter(x = s["f_mid"], y = s[metric], mode = "lines+markers", name = name))

        fig.update_layout(
            title = f"{title} {metric} vs {feature}",
            xaxis_title = feature,
            yaxis_title = metric,
            template = "simple_white"
        )
        fig.show()


# Overlaid real vs pooled model histograms, to see where the W1 comes from
def plot_marginals(real, models : dict, title : str, unit : str, clip = None):
    real = np.asarray(real)
    lo, hi = clip if clip is not None else np.quantile(real, [0.001, 0.999])

    fig = go.Figure()
    fig.add_trace(go.Histogram(x = real[(real >= lo) & (real <= hi)], name = "ISAC", histnorm = "probability density",
                               nbinsx = 150, opacity = 0.6))
    for name, draws in models.items():
        d = np.asarray(draws).ravel()
        fig.add_trace(go.Histogram(x = d[(d >= lo) & (d <= hi)], name = name, histnorm = "probability density",
                                   nbinsx = 150, opacity = 0.6))
    fig.update_layout(
        title = title,
        barmode = "overlay",
        xaxis_title = unit,
        yaxis_title = "density",
        template = "simple_white"
    )
    fig.show()
