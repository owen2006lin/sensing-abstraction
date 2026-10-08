from src.data.load import load_class, load_pos, group_split, split_pairs, FEATURE_COLS, SHARED_FEATURES
from src.components.detection import FullClassifier, MATCH_COLS
from evaluation.detection_classes import ConstantClassifier, MDNClassifier, PairMDNClassifier, MDN_PARAMS, PAIR_MDN_PARAMS, ERROR_COLS
from src.evaluation.c2st_classes import C2ST
from src.data.features import IndepFeature
from src.feature_building.feature_selector import indep_mutual_split, process_pair_features
from sklearn.metrics import roc_auc_score
import numpy as np
import polars as pl
import plotly.graph_objects as go


MODEL_PATH = "models/full_classifier.joblib"

tags, labels, features = load_class()
[X, X_val, y, y_val, t, t_val] = group_split(tags, labels, features, return_tags = True)

full_classifier = FullClassifier.load(MODEL_PATH)
train_df = pl.concat([X, y, t], how = "horizontal")
inf_df = pl.concat([X_val, t_val], how = "horizontal")
indep_preds, pair_preds = full_classifier.predict_proba(inf_df)
y_val = y_val.to_numpy().ravel()


# (1) Expected calibration error (equal mass bins, same as indep_calibration_tests)

def ece(preds, labels, n_bins):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    n = len(preds)

    order = np.argsort(preds)
    preds_sorted = preds[order]
    labels_sorted = labels[order]

    bin_edges = np.round(np.linspace(0, n, n_bins + 1)).astype(int)

    total_error = 0.0
    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        if start == end:
            continue

        avg_confidence = preds_sorted[start:end].mean()
        avg_accuracy = labels_sorted[start:end].mean()
        bin_weight = (end - start) / n

        total_error += bin_weight * abs(avg_confidence - avg_accuracy)

    return total_error


# (2) Per target marginal P(detected)
# The pair model outputs a joint over 4 classes, so collapse it into P(a), P(b) using the same
# class -> target mapping as PairClassifier.sample(). Then reuse reassemble() so the probabilities
# land on the same rows as the sampled detections (and the ground truth in inf_df)

def marginal_preds(model : FullClassifier, X, indep_preds, pair_preds):
    pair_preds = np.asarray(pair_preds)
    pair_marginals = pl.DataFrame({
        "a_detected" : pair_preds[:, 2] + pair_preds[:, 3],
        "b_detected" : pair_preds[:, 1] + pair_preds[:, 3]
    })
    indep_marginals = pl.DataFrame({"detected" : np.asarray(indep_preds)})

    df = model.reassemble(X, indep_marginals, pair_marginals)
    return df["detected"].to_numpy()


# (3) Monte Carlo null test
# Assume the model is the true data generating process. Draw synthetic labels with sample() and compute
# the ECE of the model against its own samples. Any nonzero ECE here is purely finite sample noise
# p value = fraction of null ECEs at least as large as the ECE against the real labels

def mc_null_ece(model : FullClassifier, X, labels, indep_preds, pair_preds, n_bins : int, n_sims : int = 1000, seed : int = 42):
    rng = np.random.default_rng(seed = seed)
    preds = marginal_preds(model, X, indep_preds, pair_preds)
    observed = ece(preds, labels, n_bins)

    null_eces = []
    for _ in range(n_sims):
        indep_samples, pair_samples = model.sample(indep_preds, pair_preds, rng)
        sim = model.reassemble(X, indep_samples, pair_samples)
        sim_labels = sim["detected"].cast(pl.Int8).to_numpy()
        null_eces.append(ece(preds, sim_labels, n_bins))

    null_eces = np.array(null_eces)
    # +1 so p is never exactly 0 with a finite number of sims
    p_value = (1 + np.sum(null_eces >= observed)) / (1 + n_sims)

    return observed, null_eces, p_value


def plot_mc_null(observed, null_eces, p_value, n_bins):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x = null_eces, nbinsx = 60, name = "ECE under null", opacity = 0.7))
    fig.add_vline(x = observed, line = dict(color = "firebrick", width = 2, dash = "dash"),
                  annotation_text = f"ISAC labels: {observed:.4f}", annotation_position = "top")
    fig.update_layout(
        title = f"MC Null ECE ({n_bins} bins, {len(null_eces)} sims) : p = {p_value:.4f}",
        xaxis_title = "ECE",
        yaxis_title = "count",
        template = "simple_white"
    )
    fig.show()

#--------------------------------------------------------------------------------------------------------------------#

# Example usage:
'''
observed, null_eces, p_value = mc_null_ece(full_classifier, inf_df, y_val, indep_preds, pair_preds, n_bins = 15)
plot_mc_null(observed, null_eces, p_value, n_bins = 15)
'''
#--------------------------------------------------------------------------------------------------------------------#



# (4) Branch lookup
# Same trick as marginal_preds: push integer codes through reassemble() instead of probabilities.
# Non mutual rows get -1, mutual rows get 2 * pair_index (target a) or 2 * pair_index + 1 (target b)
# so every row can be traced back to its row in pair_preds

def branch_codes(model : FullClassifier, X, n_indep : int, n_pairs : int):
    indep_codes = pl.DataFrame({"detected" : np.full(n_indep, -1)})
    pair_codes = pl.DataFrame({
        "a_detected" : 2 * np.arange(n_pairs),
        "b_detected" : 2 * np.arange(n_pairs) + 1
    })
    df = model.reassemble(X, indep_codes, pair_codes)
    return df["detected"].to_numpy()


# Bundle everything the metrics need, per point and per pair, along with scene ids for bootstrapping
# scene = (scenario_id, drop_id), same grouping as group_split

def build_eval(model : FullClassifier, X, labels, indep_preds, pair_preds):
    pair_preds = np.asarray(pair_preds)
    labels = np.asarray(labels).astype(int)
    codes = branch_codes(model, X, len(indep_preds), len(pair_preds))
    scene = X.select(pl.struct(["scenario_id", "drop_id"]).hash()).to_series().to_numpy()

    mutual = codes >= 0
    pair_idx = codes[mutual] // 2
    slot = codes[mutual] % 2

    y_a = np.zeros(len(pair_preds), dtype = int)
    y_b = np.zeros(len(pair_preds), dtype = int)
    y_a[pair_idx[slot == 0]] = labels[mutual][slot == 0]
    y_b[pair_idx[slot == 1]] = labels[mutual][slot == 1]

    # same encoding as load_pairs / split_pairs : score = 2a + b
    score = 2 * y_a + y_b
    scene_pair = np.zeros(len(pair_preds), dtype = scene.dtype)
    scene_pair[pair_idx[slot == 0]] = scene[mutual][slot == 0]

    return {
        "p_point" : marginal_preds(model, X, indep_preds, pair_preds),
        "y_point" : labels,
        "mutual" : mutual,
        "scene" : scene,
        "p_pair" : pair_preds,
        "y_pair" : np.eye(4)[score],
        "scene_pair" : scene_pair
    }


# (5) Brier score + Murphy's decomposition

def brier(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.mean((preds - labels)**2)

# Joint over (neither, only b, only a, both), sum over the 4 outcomes then average over pairs
def brier_multiclass(probs, onehot):
    return np.mean(np.sum((probs - onehot)**2, axis = 1))

# Equal mass bins, same as murphy_decomp in indep_calibration_tests
# rel - res + unc only approximates the brier score since within bin variance is dropped
def murphy_decomp(preds, labels, n_bins):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    N = len(preds)

    order = np.argsort(preds)
    preds_sorted = preds[order]
    labels_sorted = labels[order]
    o = labels.mean()

    rel = 0
    res = 0
    unc = o * (1 - o)

    # equal mass bins, but tied predictions are forced into the same bin. Otherwise a constant predictor
    # gets split into arbitrary chunks and picks up fake reliability / resolution
    bin_edges = np.round(np.linspace(0, N, n_bins + 1)).astype(int)
    bin_of_pos = np.searchsorted(bin_edges, np.arange(N), side = "right") - 1
    _, first, inverse = np.unique(preds_sorted, return_index = True, return_inverse = True)
    bins = bin_of_pos[first][inverse]

    for i in np.unique(bins):
        mask = bins == i
        n = mask.sum()
        mean_p = preds_sorted[mask].mean()
        prob = labels_sorted[mask].mean()

        rel += n * (mean_p - prob)**2
        res += n * (prob - o)**2

    rel = rel / N
    res = res / N

    return {
        "brier_score" : brier(preds, labels),
        "reliability" : rel,
        "resolution" : res,
        "uncertainty" : unc,
        "reconstructed_brier" : rel - res + unc,  # should ≈ brier_score
    }


# (6) AUROC

def auroc(preds, labels):
    labels = np.asarray(labels)
    # undefined if a resample/bin only has one class
    if labels.min() == labels.max():
        return np.nan
    return roc_auc_score(labels, preds)


#--------------------------------------------------------------------------------------------------------------------#

# Branch metrics : non mutual points, mutual pairs (joint + per point marginals), and everything pooled
BRANCHES = ["non_mutual", "mutual", "pooled"]

def branch_metrics(ev, n_bins, point_idx = None, pair_idx = None):
    if point_idx is None:
        point_idx = np.arange(len(ev["p_point"]))
    if pair_idx is None:
        pair_idx = np.arange(len(ev["p_pair"]))

    p = ev["p_point"][point_idx]
    y = ev["y_point"][point_idx]
    mutual = ev["mutual"][point_idx]
    masks = {"non_mutual" : ~mutual, "mutual" : mutual, "pooled" : np.ones_like(mutual)}

    out = {}
    for branch, mask in masks.items():
        decomp = murphy_decomp(p[mask], y[mask], n_bins)
        out[(branch, "brier")] = decomp["brier_score"]
        out[(branch, "reliability")] = decomp["reliability"]
        out[(branch, "resolution")] = decomp["resolution"]
        out[(branch, "uncertainty")] = decomp["uncertainty"]
        out[(branch, "auroc")] = auroc(p[mask], y[mask])

    out[("mutual", "brier_joint")] = brier_multiclass(ev["p_pair"][pair_idx], ev["y_pair"][pair_idx])
    return out


# Bootstrap by scene : resample whole scenes with replacement, since points within a scene are correlated
def scene_bootstrap(ev, n_bins, n_boot : int = 1000, alpha : float = 0.05, seed : int = 42):
    rng = np.random.default_rng(seed = seed)
    scenes = np.unique(ev["scene"])
    rows_by_scene = {s : np.flatnonzero(ev["scene"] == s) for s in scenes}
    pairs_by_scene = {s : np.flatnonzero(ev["scene_pair"] == s) for s in scenes}

    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(scenes, size = len(scenes), replace = True)
        point_idx = np.concatenate([rows_by_scene[s] for s in chosen])
        pair_idx = np.concatenate([pairs_by_scene[s] for s in chosen])
        boots.append(branch_metrics(ev, n_bins, point_idx, pair_idx))

    keys = boots[0].keys()
    ci = {}
    for k in keys:
        vals = np.array([b[k] for b in boots])
        ci[k] = (np.nanquantile(vals, alpha / 2), np.nanquantile(vals, 1 - alpha / 2))
    return ci


# Constant rate baseline : per branch training detection rate for points, training class frequencies for pairs
def constant_preds(train_df, indep_preds, pair_preds):
    _, non_mutual = indep_mutual_split(train_df, MATCH_COLS)
    rate = non_mutual["detected"].mean()

    pairs = process_pair_features(train_df, MATCH_COLS)
    _, pair_labels, _ = split_pairs(pairs, FEATURE_COLS + SHARED_FEATURES)
    freqs = np.bincount(pair_labels["score"].to_numpy(), minlength = 4) / len(pair_labels)

    return np.full(len(indep_preds), rate), np.tile(freqs, (len(pair_preds), 1))


# models : {name : (indep_preds, pair_preds)}, so any extra baselines just need to output the same format
def score_models(model : FullClassifier, X, labels, models : dict, n_bins : int = 15, n_boot : int = 1000, seed : int = 42):
    rows = []
    for name, (i_preds, p_preds) in models.items():
        ev = build_eval(model, X, labels, i_preds, p_preds)
        point = branch_metrics(ev, n_bins)
        ci = scene_bootstrap(ev, n_bins, n_boot, seed = seed)

        for (branch, metric), val in point.items():
            lo, hi = ci[(branch, metric)]
            rows.append({"model" : name, "branch" : branch, "metric" : metric, "value" : val, "lo" : lo, "hi" : hi})

    return pl.DataFrame(rows)


# Wide table, one per metric : rows = models, columns = branches, cells = "value [lo, hi]"
def print_tables(scores : pl.DataFrame):
    scores = scores.with_columns(
        pl.format("{} [{}, {}]",
                  pl.col("value").round(4), pl.col("lo").round(4), pl.col("hi").round(4)).alias("cell")
    )
    for metric in scores["metric"].unique(maintain_order = True):
        table = (scores
                    .filter(pl.col("metric") == metric)
                    .pivot(on = "branch", index = "model", values = "cell"))
        print(f"\n{metric}")
        print(table)


# (7) Appendix : AUROC and Brier per bin of a covariate (pooled per point marginals)
SLICE_FEATURES = ["bistatic_range_m", "az_off_boresight"]

def slice_metrics(model : FullClassifier, X, labels, models : dict, feature : str, n_bins : int = 8):
    col = X[feature].to_numpy()
    valid = ~np.isnan(col)
    order = np.argsort(col[valid])
    chunks = np.array_split(np.flatnonzero(valid)[order], n_bins)

    rows = []
    for name, (i_preds, p_preds) in models.items():
        ev = build_eval(model, X, labels, i_preds, p_preds)
        for idx in chunks:
            p, y = ev["p_point"][idx], ev["y_point"][idx]
            rows.append({
                "model" : name,
                "f_mid" : col[idx].mean(),
                "n" : len(idx),
                "brier" : brier(p, y),
                "auroc" : auroc(p, y)
            })
    return pl.DataFrame(rows)


def plot_slices(slices : pl.DataFrame, feature : str):
    for metric in ["auroc", "brier"]:
        fig = go.Figure()
        for name in slices["model"].unique(maintain_order = True):
            s = slices.filter(pl.col("model") == name)
            fig.add_trace(go.Scatter(x = s["f_mid"], y = s[metric], mode = "lines+markers", name = name))

        fig.update_layout(
            title = f"{metric} vs {feature}",
            xaxis_title = feature,
            yaxis_title = metric,
            template = "simple_white"
        )
        fig.show()

#--------------------------------------------------------------------------------------------------------------------#

# (8) Baselines B1 (ConstantClassifier) and B4 (MDNClassifier) from src.evaluation.classes
# Both output a per point P(detected) with no pair coupling, so convert them into the FullClassifier
# (indep_preds, pair_preds) format with the pair joint = product of marginals (independence)

def to_branch_preds(model : FullClassifier, X, point_preds, n_indep : int, n_pairs : int):
    point_preds = np.asarray(point_preds)
    codes = branch_codes(model, X, n_indep, n_pairs)
    mutual = codes >= 0
    pair_idx = codes[mutual] // 2
    slot = codes[mutual] % 2

    p_a = np.zeros(n_pairs)
    p_b = np.zeros(n_pairs)
    p_a[pair_idx[slot == 0]] = point_preds[mutual][slot == 0]
    p_b[pair_idx[slot == 1]] = point_preds[mutual][slot == 1]

    # same encoding as split_pairs : 0 neither, 1 only b, 2 only a, 3 both
    pair_preds = np.stack([(1 - p_a) * (1 - p_b), (1 - p_a) * p_b, p_a * (1 - p_b), p_a * p_b], axis = 1)
    return point_preds[~mutual], pair_preds


# Baselines also learn the error distribution, so attach the load_pos() errors to the detection labels (null if missed)
def baseline_labels(labels, tags, error_cols = ERROR_COLS):
    pos_tags, pos_labels, _ = load_pos()
    errors = pl.concat([pos_tags, pos_labels.select(error_cols)], how = "horizontal")

    return (pl.concat([labels, tags], how = "horizontal")
              .join(errors, on = ["scenario_id", "drop_id", "target_id"], how = "left", maintain_order = "left")
              .select(["detected", *error_cols]))

#--------------------------------------------------------------------------------------------------------------------#

# (9) C2ST : classifier two sample test
# Real rows = (features, ISAC detected), synthetic rows = (same features, detected sampled from the model)
# nn_detected = the mutual partner's detected (null for non mutual points) so the discriminator can also catch
# a wrong joint, eg B4 / B1 treating mutual pairs as independent
# Null : train the same C2ST on two independent synthetic draws, AUC under "model is the truth"

C2ST_FEATURES = [f.value for f in IndepFeature]

# Row index of each point's mutual partner (-1 if non mutual), using the branch_codes lookup
def partner_index(model : FullClassifier, X, n_indep : int, n_pairs : int):
    codes = branch_codes(model, X, n_indep, n_pairs)
    mutual = np.flatnonzero(codes >= 0)

    row_of_code = np.zeros(2 * n_pairs, dtype = int)
    row_of_code[codes[mutual]] = mutual

    partner = np.full(len(codes), -1)
    partner[mutual] = row_of_code[codes[mutual] ^ 1]   # 2i <-> 2i + 1
    return partner


def c2st_frame(X, detected, partner, feature_cols = C2ST_FEATURES):
    detected = np.asarray(detected).astype(float)
    nn_detected = np.where(partner >= 0, detected[partner], np.nan)
    return X.select(feature_cols).with_columns(
        pl.Series("detected", detected),
        pl.Series("nn_detected", nn_detected).fill_nan(None)
    )


def sample_detected(model : FullClassifier, X, indep_preds, pair_preds, rng):
    indep_samples, pair_samples = model.sample(indep_preds, pair_preds, rng)
    sim = model.reassemble(X, indep_samples, pair_samples)
    return sim["detected"].cast(pl.Int8).to_numpy()


def run_c2st(model : FullClassifier, X, labels, indep_preds, pair_preds, n_null : int = 20, n_folds : int = 5, seed : int = 42):
    rng = np.random.default_rng(seed = seed)
    partner = partner_index(model, X, len(indep_preds), len(pair_preds))
    groups = X.select(pl.struct(["scenario_id", "drop_id"]).hash()).to_series().to_numpy()

    real = c2st_frame(X, labels, partner)
    synth = c2st_frame(X, sample_detected(model, X, indep_preds, pair_preds, rng), partner)
    c2st = C2ST(n_folds = n_folds).fit(real, synth, groups)

    null_aucs = []
    for _ in range(n_null):
        synth_a = c2st_frame(X, sample_detected(model, X, indep_preds, pair_preds, rng), partner)
        synth_b = c2st_frame(X, sample_detected(model, X, indep_preds, pair_preds, rng), partner)
        null_aucs.append(C2ST(n_folds = n_folds).fit(synth_a, synth_b, groups).auc)

    null_aucs = np.array(null_aucs)
    # +1 so p is never exactly 0 with a finite number of sims
    p_value = (1 + np.sum(null_aucs >= c2st.auc)) / (1 + n_null)
    return c2st, null_aucs, p_value


# models : {name : (indep_preds, pair_preds)}, same format as score_models
def c2st_models(model : FullClassifier, X, labels, models : dict, n_null : int = 20, n_folds : int = 5, seed : int = 42):
    rows = []
    results = {}
    for name, (i_preds, p_preds) in models.items():
        c2st, null_aucs, p_value = run_c2st(model, X, labels, i_preds, p_preds, n_null, n_folds, seed)
        results[name] = (c2st, null_aucs, p_value)
        print(f"C2ST {name} : auc {c2st.auc:.4f}, null mean {null_aucs.mean():.4f}, p = {p_value:.4f}")
        rows.append({
            "model" : name,
            "auc" : c2st.auc,
            "acc" : c2st.acc,
            "null_mean" : null_aucs.mean(),
            "null_95" : np.quantile(null_aucs, 0.95),
            "p_value" : p_value
        })
    return pl.DataFrame(rows), results


def plot_c2st_null(c2st : C2ST, null_aucs, p_value, name : str):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x = null_aucs, nbinsx = 30, name = "AUC under null", opacity = 0.7))
    fig.add_vline(x = c2st.auc, line = dict(color = "firebrick", width = 2, dash = "dash"),
                  annotation_text = f"real vs {name}: {c2st.auc:.4f}", annotation_position = "top")
    fig.update_layout(
        title = f"C2ST {name} ({len(null_aucs)} null sims) : p = {p_value:.4f}",
        xaxis_title = "out of fold AUC",
        yaxis_title = "count",
        template = "simple_white"
    )
    fig.show()

# Which features the discriminator leans on, eg detected / nn_detected interactions point at the joint
def plot_c2st_importance(c2st : C2ST, name : str, top_k : int = 15):
    imp = c2st.importance.head(top_k)
    fig = go.Figure(data = [go.Bar(x = imp["gain"], y = imp["feature"], orientation = "h")])
    fig.update_layout(
        title = f"C2ST {name} feature importance (gain)",
        xaxis_title = "gain",
        yaxis = dict(autorange = "reversed"),
        template = "simple_white"
    )
    fig.show()

#--------------------------------------------------------------------------------------------------------------------#

# Example usage:

# ---- Setup : fit baselines B1, B4 and collect every model in (indep_preds, pair_preds) format ----
y_base = baseline_labels(y, t)
b1 = ConstantClassifier().fit(X, y_base)
b1.save("src/evaluation/models/constant.joblib")

# inner group split of the train set for early stopping, so the eval set stays held out
[X_b4, X_b4_val, y_b4, y_b4_val] = group_split(t, y_base, X)
b4 = MDNClassifier().fit(X_b4, y_b4, X_b4_val, y_b4_val)
b4.save("src/evaluation/models/MDN.joblib")

b1_indep_preds, b1_pair_preds = to_branch_preds(full_classifier, inf_df, b1.predict_proba(inf_df), len(indep_preds), len(pair_preds))
b4_indep_preds, b4_pair_preds = to_branch_preds(full_classifier, inf_df, b4.predict_proba(inf_df), len(indep_preds), len(pair_preds))

# B5 : MDN with the same indep / mutual split as full_classifier, inner group split for early stopping
[train_b5, val_b5, _, _] = group_split(t, y, train_df)
b5 = FullClassifier(models = [MDNClassifier(), PairMDNClassifier()], params = [MDN_PARAMS, PAIR_MDN_PARAMS], callbacks = [None, None]).fit(train_b5, val_b5)
b5_indep_preds, b5_pair_preds = b5.predict_proba(inf_df)

models = {
    "constant rate" : constant_preds(train_df, indep_preds, pair_preds),
    "B1" : (b1_indep_preds, b1_pair_preds),
    "B4" : (b4_indep_preds, b4_pair_preds),
    "B5" : (b5_indep_preds, b5_pair_preds),
    "full classifier" : (indep_preds, pair_preds)
}

# ---- (5), (6) Brier + Murphy decomposition, AUROC, per branch with scene bootstrap CIs ----
scores = score_models(full_classifier, inf_df, y_val, models, n_bins = 15, n_boot = 1000)
print_tables(scores)

# ---- (3) MC null ECE ----
# MC null ECE for the baselines. full_classifier is only used for its samplers + reassemble(), which are generic
# bernoulli / categorical draws, so sampling the independent joint is the same as B1 / B4 .sample()
for name in ["B1", "B4", "full classifier"]:
    observed, null_eces, p_value = mc_null_ece(full_classifier, inf_df, y_val, *models[name], n_bins = 15)
    plot_mc_null(observed, null_eces, p_value, n_bins = 15)

# ---- (7) Appendix : AUROC / Brier sliced by range and angle ----

for feature in SLICE_FEATURES:
    slices = slice_metrics(full_classifier, inf_df, y_val, models, feature, n_bins = 8)
    plot_slices(slices, feature)


# ---- (9) C2ST : real vs synthetic discriminator, null from synthetic vs synthetic ----
'''
c2st_table, c2st_results = c2st_models(full_classifier, inf_df, y_val, models, n_null = 20)
print(c2st_table)
for name, (c2st, null_aucs, p_value) in c2st_results.items():
    plot_c2st_null(c2st, null_aucs, p_value, name)
    plot_c2st_importance(c2st, name)
'''
    #--------------------------------------------------------------------------------------------------------------------#


