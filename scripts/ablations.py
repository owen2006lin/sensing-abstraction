import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import io
import contextlib
import numpy as np
import polars as pl
import lightgbm as lgb

from src.components.base import Component
from src.components.detection import FullClassifier, IndepClassifier, PairClassifier, MATCH_COLS
from src.components.position import FullPositionModel
from src.components.range.distributions import label_regime, REGION_NAMES
from src.data.features import PosFeature
from src.data.load import load_class, load_pos, group_split
from src.evaluation.detection_tests import build_eval, brier_multiclass, auroc, score_models as score_detection
from src.evaluation.error_metrics import scene_ids, crps_ensemble, w1, tile_frame, to_draws
from src.evaluation.range_tests import sample_range, climatology_draws
from src.feature_building.feature_selector import indep_mutual_split, process_pair_inference

# Three quick ablations, results saved to results/ablations/
#   joint    : joint vs independent detection for mutual pairs (no training)
#   features : detection performance with each feature group removed (retrains FullClassifier per group)
#   range    : range model without the regime classifier (direct quantile GBM from features to range error)
# Run from the repo root after scripts/train.py : python scripts/ablations.py [joint] [features] [range]
OUT = Path("results/ablations")
OUT.mkdir(parents = True, exist_ok = True)
DETECTION_PATH = "models/full_classifier.joblib"
POSITION_PATH = "models/position_model.joblib"
N_BOOT = 1000
SEED = 42


def split(tags, labels, features):
    # same two step split as scripts/train.py : outer split holds out test, inner split gives early stopping val
    [X, X_test, y, y_test, t, t_test] = group_split(tags, labels, features, return_tags = True)
    [X, X_val, y, y_val, t, t_val] = group_split(t, y, X, return_tags = True)
    return (X, y, t), (X_val, y_val, t_val), (X_test, y_test, t_test)


# Paired scene bootstrap : resample whole scenes, recompute stat_fn on the same resample for every model
# With pair_scene, stat_fn(idx, pair_idx) also gets the pair rows of the same resampled scenes
def paired_bootstrap(stat_fn, scene, pair_scene = None, n_boot = N_BOOT, seed = SEED):
    rng = np.random.default_rng(seed)
    scenes = np.unique(scene)
    rows = {s : np.flatnonzero(scene == s) for s in scenes}
    if pair_scene is not None:
        pair_rows = {s : np.flatnonzero(pair_scene == s) for s in scenes}
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(scenes, size = len(scenes), replace = True)
        idx = np.concatenate([rows[s] for s in chosen])
        if pair_scene is None:
            boots.append(stat_fn(idx))
        else:
            boots.append(stat_fn(idx, np.concatenate([pair_rows[s] for s in chosen])))
    boots = np.array(boots)
    return np.nanquantile(boots, 0.025, axis = 0), np.nanquantile(boots, 0.975, axis = 0)


def write_summary(name, title, body):
    (OUT / f"{name}.md").write_text(f"# {title}\n\n```\n{body}```\n")
    print(body)


def capture(fn):
    with pl.Config(tbl_rows = -1, tbl_cols = -1, tbl_width_chars = 200, fmt_str_lengths = 40), io.StringIO() as buf:
        with contextlib.redirect_stdout(buf):
            fn()
        return buf.getvalue()


def load_detection_test():
    (X, y, t), (X_val, y_val, t_val), (X_test, y_test, t_test) = split(*load_class())
    train_df = pl.concat([X, y, t], how = "horizontal")
    val_df = pl.concat([X_val, y_val, t_val], how = "horizontal")
    inf_df = pl.concat([X_test, t_test], how = "horizontal")
    return train_df, val_df, inf_df, y_test.to_numpy().ravel()


#==========================================(1) Joint vs independent detection==========================================#
# Collapse the pair model's 4 class joint to marginals p_A = P(2) + P(3), p_B = P(1) + P(3), and rebuild the joint
# as if the targets were independent. Class order is score = 2a + b : (neither, only b, only a, both).
# Marginals are identical by construction, so any Brier difference is purely from modelling the dependence

def independent_joint(pair_preds):
    p = np.asarray(pair_preds)
    p_a, p_b = p[:, 2] + p[:, 3], p[:, 1] + p[:, 3]
    return np.column_stack([(1 - p_a) * (1 - p_b), (1 - p_a) * p_b, p_a * (1 - p_b), p_a * p_b])


# Detection correlation between the two targets of a pair, from a (n, 4) joint (rows can be one hot labels)
def pair_corr(joint):
    joint = np.asarray(joint).mean(axis = 0)
    p_a, p_b, p_ab = joint[2] + joint[3], joint[1] + joint[3], joint[3]
    return (p_ab - p_a * p_b) / np.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))


def per_pair_brier(probs, onehot):
    return np.sum((probs - onehot)**2, axis = 1)


def per_pair_logloss(probs, onehot):
    return -np.log(np.clip(np.sum(probs * onehot, axis = 1), 1e-12, None))


def ablation_joint():
    _, _, inf_df, y_true = load_detection_test()
    model = FullClassifier.load(DETECTION_PATH)
    indep_preds, pair_preds = model.predict_proba(inf_df)
    ev = build_eval(model, inf_df, y_true, indep_preds, pair_preds)

    joint, onehot, scene = ev["p_pair"], ev["y_pair"], ev["scene_pair"]
    indep = independent_joint(joint)
    b_joint, b_indep = per_pair_brier(joint, onehot), per_pair_brier(indep, onehot)
    l_joint, l_indep = per_pair_logloss(joint, onehot), per_pair_logloss(indep, onehot)

    # sanity check : marginals really are identical
    assert np.allclose(independent_joint(indep), indep)

    def stats(idx):
        return np.array([b_joint[idx].mean(), b_indep[idx].mean(), b_indep[idx].mean() - b_joint[idx].mean(),
                         l_indep[idx].mean() - l_joint[idx].mean()])
    lo, hi = paired_bootstrap(stats, scene)
    point = stats(np.arange(len(joint)))
    names = ["brier joint", "brier independent", "brier diff (indep - joint)", "logloss diff (indep - joint)"]
    table = pl.DataFrame({"metric" : names, "value" : point, "lo" : lo, "hi" : hi})

    # where does the dependence matter : brier gain by quartile of normalised range separation between the pair
    mutual, _ = indep_mutual_split(inf_df, MATCH_COLS)
    sep = process_pair_inference(mutual, MATCH_COLS)["nn_norm_range_sep"].cast(pl.Float64).to_numpy()
    q = np.quantile(sep, [0.25, 0.5, 0.75])
    bins = np.searchsorted(q, sep)
    slices = pl.DataFrame([{
        "sep_quartile" : k + 1,
        "sep_mid" : float(np.median(sep[bins == k])),
        "n_pairs" : int((bins == k).sum()),
        "real_corr" : pair_corr(onehot[bins == k]),
        "model_corr" : pair_corr(joint[bins == k]),
        "brier_joint" : b_joint[bins == k].mean(),
        "brier_indep" : b_indep[bins == k].mean(),
        "rel_gain_pct" : 100 * (b_indep[bins == k].mean() - b_joint[bins == k].mean()) / b_indep[bins == k].mean(),
    } for k in range(4)])

    table.write_csv(OUT / "joint_vs_indep.csv")
    slices.write_csv(OUT / "joint_vs_indep_by_sep.csv")

    def show():
        print(f"test set : {len(joint)} mutual pairs")
        print(f"detection correlation within pairs : real {pair_corr(onehot):.3f}, pair model {pair_corr(joint):.3f}, "
              f"independent 0 by construction\n")
        print("4 class Brier / log loss, value [95% paired scene bootstrap CI]")
        print(table)
        print("\nby quartile of nn_norm_range_sep (1 = closest pairs)")
        print(slices)
    write_summary("joint_vs_indep", "Ablation : joint vs independent detection for mutual pairs", capture(show))


#==========================================(2) Feature groups==========================================#
# Groups follow the comments in src/data/features.py. The pair model sees per target columns with _a / _b suffixes,
# so a column is dropped when its unsuffixed name is in the group

FEATURE_GROUPS = {
    "geometry" : ["bistatic_range_m", "log_R", "az_off_boresight", "rx_elevation_deg", "radial_velocity",
                  "abs_bistatic_doppler_hz", "speed"],
    "quantization" : ["abs_frac_u", "abs_frac_v"],
    "snr" : ["n_mpc", "rss_max", "rss_total_db", "k_factor_db", "dom_excess_delay", "rms_delay_spread",
             "aoa_spread_w", "zoa_spread_w", "first_rss_minus_dom"],
    "interference" : ["n_targets", "nn_norm_range_sep", "nn_norm_ang_sep"],
}
QUIET_CALLBACKS = [lgb.early_stopping(stopping_rounds = 200, verbose = False)]


def in_group(col, group):
    base = col[:-2] if col.endswith(("_a", "_b")) else col
    return col in group or base in group


# Wraps a detection member and hides a set of columns from it, in fit and predict_proba
class DropFeatures(Component):
    def __init__(self, model, drop):
        self.model = model
        self.drop = list(drop)

    def _keep(self, X):
        return X.select([c for c in X.columns if not in_group(c, self.drop)])

    def fit(self, X, y, X_val = None, y_val = None, params = None, callbacks = None):
        X_val = None if X_val is None else self._keep(X_val)
        self.model.fit(self._keep(X), y, X_val, y_val, params, callbacks)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._keep(X))

    def sample(self, X, rng):
        return self.model.sample(X, rng)


def fit_without(train_df, val_df, drop):
    model = FullClassifier(models = [DropFeatures(IndepClassifier(), drop), DropFeatures(PairClassifier(), drop)],
                           callbacks = [QUIET_CALLBACKS, QUIET_CALLBACKS])
    return model.fit(train_df, val_df)


def ablation_features(groups):
    train_df, val_df, inf_df, y_true = load_detection_test()
    variants = {"all features" : []}
    variants.update({f"no {g}" : FEATURE_GROUPS[g] for g in groups})
    if "snr" in groups:
        # snr alone : everything except snr dropped, to see how much the channel features carry on their own
        variants["snr only"] = [c for g, cols in FEATURE_GROUPS.items() if g != "snr" for c in cols]

    preds, base = {}, None
    for name, drop in variants.items():
        print(f"fitting : {name}")
        model = fit_without(train_df, val_df, drop)
        base = base or model    # any fitted FullClassifier works for build_eval (only uses reassemble)
        preds[name] = model.predict_proba(inf_df)

    scores = score_detection(base, inf_df, y_true, preds, n_boot = N_BOOT)
    scores.write_csv(OUT / "feature_groups.csv")

    # paired difference vs all features on the same scene resamples : pooled per point Brier, AUROC, 4 class Brier
    evs = {n : build_eval(base, inf_df, y_true, *p) for n, p in preds.items()}
    ref = evs["all features"]
    rows = []
    for name, ev in evs.items():
        if name == "all features":
            continue
        def diff(idx, pidx, ev = ev):
            y = ref["y_point"][idx]
            return np.array([
                np.mean((ev["p_point"][idx] - y)**2) - np.mean((ref["p_point"][idx] - y)**2),
                auroc(ev["p_point"][idx], y) - auroc(ref["p_point"][idx], y),
                brier_multiclass(ev["p_pair"][pidx], ev["y_pair"][pidx])
                - brier_multiclass(ref["p_pair"][pidx], ref["y_pair"][pidx]),
            ])
        point = diff(np.arange(len(y_true)), np.arange(len(ref["p_pair"])))
        lo, hi = paired_bootstrap(diff, ref["scene"], ref["scene_pair"])
        for m, v, l, h in zip(["d_brier_pooled", "d_auroc_pooled", "d_brier_joint"], point, lo, hi):
            rows.append({"variant" : name, "metric" : m, "value" : v, "lo" : l, "hi" : h})
    diffs = pl.DataFrame(rows)
    diffs.write_csv(OUT / "feature_groups_diff.csv")

    def show():
        print(f"test set : {len(y_true)} targets. Each variant retrains FullClassifier (same split, params, early stopping)\n")
        cell = pl.format("{} [{}, {}]", pl.col("value").round(4), pl.col("lo").round(4), pl.col("hi").round(4))
        wide = (scores.filter(pl.col("metric").is_in(["brier", "auroc", "brier_joint"]))
                      .with_columns(cell.alias("cell"), (pl.col("branch") + " " + pl.col("metric")).alias("col"))
                      .pivot(on = "col", index = "model", values = "cell"))
        print("value [95% scene bootstrap CI]")
        print(wide)
        print("\nchange vs all features (positive brier / negative auroc = worse), [95% paired scene bootstrap CI]")
        print(diffs.with_columns(cell.alias("cell")).pivot(on = "metric", index = "variant", values = "cell"))
    write_summary("feature_groups", "Ablation : detection feature groups", capture(show))


#==========================================(3) Range model without the regime classifier==========================================#
# Direct model : one LightGBM quantile regressor per level, PosFeature -> signed range error, no regimes.
# Same recipe as the B2 baseline in src/evaluation/position_classes.py, minus the log (range error is signed).
# Sampling is inverse CDF between the predicted quantiles, clamped to the outer levels

TAUS = np.array([0.005, 0.01, 0.025, *np.round(np.arange(0.05, 0.951, 0.05), 2), 0.975, 0.99, 0.995])
DIRECT_PARAMS = {
    "objective" : "quantile", "metric" : "quantile", "learning_rate" : 0.05, "num_leaves" : 31,
    "min_data_in_leaf" : 100, "feature_fraction" : 0.8, "bagging_fraction" : 0.8, "bagging_freq" : 1,
    "lambda_l2" : 1.0, "verbosity" : -1, "seed" : SEED, "device" : "cpu", "num_threads" : 6,
}
FEATURES = [f.value for f in PosFeature]
ERROR_COL = "range_error_m"
N_DRAWS = 30
N_TRAIN_EVAL = 9000    # train rows scored for the train / test gap (regime model sampling is slow)


class DirectRangeModel:
    def __init__(self, params = DIRECT_PARAMS, taus = TAUS, num_boost_round = 2000, early_stopping = 50):
        self.params, self.taus = params, taus
        self.num_boost_round, self.early_stopping = num_boost_round, early_stopping

    def fit(self, X, y, X_val, y_val):
        dtrain = lgb.Dataset(X.select(FEATURES).to_numpy().astype(float), label = y, free_raw_data = False)
        dval = lgb.Dataset(X_val.select(FEATURES).to_numpy().astype(float), label = y_val, reference = dtrain)
        self.models = [lgb.train({**self.params, "alpha" : tau}, dtrain, self.num_boost_round, valid_sets = [dval],
                                 callbacks = [lgb.early_stopping(self.early_stopping, verbose = False)])
                       for tau in self.taus]
        return self

    def quantiles(self, X):
        X = X.select(FEATURES).to_numpy().astype(float)
        return np.sort(np.column_stack([m.predict(X, num_iteration = m.best_iteration) for m in self.models]), axis = 1)

    def draws(self, X, n_draws, rng):
        q = self.quantiles(X)
        u = np.clip(rng.uniform(size = (len(q), n_draws)), self.taus[0], self.taus[-1])
        hi = np.clip(np.searchsorted(self.taus, u), 1, len(self.taus) - 1)
        lo = hi - 1
        w = (u - self.taus[lo]) / (self.taus[hi] - self.taus[lo])
        rows = np.arange(len(q))[:, None]
        return (1 - w) * q[rows, lo] + w * q[rows, hi]


def pinball(q, y, taus):
    d = y[:, None] - q
    return np.maximum(taus * d, (taus - 1) * d)    # (n, n_taus)


def ablation_range():
    (X, y, t), (X_val, y_val, _), (X_test, y_test, t_test) = split(*load_pos())
    y_train, y_v, y_true = (d[ERROR_COL].to_numpy() for d in (y, y_val, y_test))
    regime_train = label_regime(y)["region_txt"].to_numpy()
    regime_test = label_regime(y_test)["region_txt"].to_numpy()

    print("fitting direct quantile model")
    direct = DirectRangeModel().fit(X, y_train, X_val, y_v)
    range_model = FullPositionModel.load(POSITION_PATH).range_model

    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(len(X), size = min(N_TRAIN_EVAL, len(X)), replace = False))
    X_tr, y_tr, reg_tr = X[sub], y_train[sub], regime_train[sub]

    print("sampling")
    draws = {
        "climatology" : {"train" : climatology_draws(y_train, len(sub), N_DRAWS, rng),
                         "test" : climatology_draws(y_train, len(y_true), N_DRAWS, rng)},
        "regime model" : {"train" : sample_range(range_model, X_tr, N_DRAWS, rng),
                          "test" : sample_range(range_model, X_test, N_DRAWS, rng)},
        "direct (no regimes)" : {"train" : direct.draws(X_tr, N_DRAWS, rng),
                                 "test" : direct.draws(X_test, N_DRAWS, rng)},
    }
    real = {"train" : y_tr, "test" : y_true}
    regime = {"train" : reg_tr, "test" : regime_test}

    # CRPS by regime group, train vs test. Rare = every non bulk regime (sidebands, far, near)
    rows = []
    for name, d in draws.items():
        for part in ("train", "test"):
            c = crps_ensemble(real[part], d[part])
            bulk = regime[part] == "bulk"
            far = np.isin(regime[part], ["far+", "far-"])
            rows.append({"model" : name, "split" : part, "crps_all" : c.mean(), "crps_bulk" : c[bulk].mean(),
                         "crps_rare" : c[~bulk].mean(), "crps_far" : c[far].mean(),
                         "w1" : w1(real[part], d[part])})
    crps_table = pl.DataFrame(rows)

    # train / test gap ratio per group : > 1 means the model fits training data better than unseen data
    gap = (crps_table.pivot(on = "split", index = "model", values = ["crps_all", "crps_bulk", "crps_rare", "crps_far"])
           .select("model", *[(pl.col(f"{m}_test") / pl.col(f"{m}_train")).round(3).alias(f"{m} test/train")
                              for m in ["crps_all", "crps_bulk", "crps_rare", "crps_far"]]))

    # test CRPS by regime with paired scene bootstrap CI on (direct - regime model)
    scene = scene_ids(t_test)
    c_reg = crps_ensemble(y_true, draws["regime model"]["test"])
    c_dir = crps_ensemble(y_true, draws["direct (no regimes)"]["test"])
    groups = {"all" : np.ones(len(y_true), bool), "bulk" : regime_test == "bulk", "rare" : regime_test != "bulk",
              **{r : regime_test == r for r in REGION_NAMES[1:]}}
    def diff(idx):
        return np.array([c_dir[idx][g[idx]].mean() - c_reg[idx][g[idx]].mean() if g[idx].any() else np.nan
                         for g in groups.values()])
    lo, hi = paired_bootstrap(diff, scene)
    point = diff(np.arange(len(y_true)))
    by_regime = pl.DataFrame({
        "group" : list(groups), "n_test" : [int(g.sum()) for g in groups.values()],
        "frac_train" : [float(np.mean(regime_train == r)) if r not in ("all", "rare") else
                        (1.0 if r == "all" else float(np.mean(regime_train != "bulk"))) for r in groups],
        "crps_regime" : [c_reg[g].mean() for g in groups.values()],
        "crps_direct" : [c_dir[g].mean() for g in groups.values()],
        "diff (direct - regime)" : point, "lo" : lo, "hi" : hi,
    })

    # quantile calibration and pinball gap of the direct model : central vs tail levels
    q_tr, q_te = direct.quantiles(X_tr), direct.quantiles(X_test)
    pb_tr, pb_te = pinball(q_tr, y_tr, TAUS).mean(axis = 0), pinball(q_te, y_true, TAUS).mean(axis = 0)
    quant = pl.DataFrame({
        "tau" : TAUS,
        "best_iter" : [m.best_iteration for m in direct.models],
        "cover_train" : (y_tr[:, None] <= q_tr).mean(axis = 0),
        "cover_test" : (y_true[:, None] <= q_te).mean(axis = 0),
        "pinball_test/train" : pb_te / pb_tr,
    }).filter(pl.col("tau").is_in([0.005, 0.01, 0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975, 0.99, 0.995]))

    # how often does each model put mass far out (|error| > 1.85 m, the far regimes), vs the real rate
    tail = pl.DataFrame([{"model" : n, "P(|err| > 1.85) test" : float(np.mean(np.abs(d["test"]) > 1.85)),
                          "P(err > 1.85) test" : float(np.mean(d["test"] > 1.85)),
                          "P(err < -2.25) test" : float(np.mean(d["test"] < -2.25))} for n, d in draws.items()]
                        + [{"model" : "ISAC (real)", "P(|err| > 1.85) test" : float(np.mean(np.abs(y_true) > 1.85)),
                            "P(err > 1.85) test" : float(np.mean(y_true > 1.85)),
                            "P(err < -2.25) test" : float(np.mean(y_true < -2.25))}])

    crps_table.write_csv(OUT / "range_crps.csv")
    by_regime.write_csv(OUT / "range_by_regime.csv")
    quant.write_csv(OUT / "range_direct_quantiles.csv")
    tail.write_csv(OUT / "range_tails.csv")

    def show():
        print(f"test set : {len(y_true)} targets, train subsample : {len(sub)} targets, {N_DRAWS} draws per target\n")
        print("CRPS (m) by split and regime group")
        print(crps_table)
        print("\ntrain / test gap (test CRPS / train CRPS)")
        print(gap)
        print("\ntest CRPS by regime, diff = direct - regime model [95% paired scene bootstrap CI]")
        print(by_regime)
        print("\ndirect model : quantile coverage (P(y <= q_tau)) and pinball test/train ratio")
        print(quant)
        print("\ntail mass on the test set")
        print(tail)
    write_summary("range_no_regime", "Ablation : range model without the regime classifier", capture(show))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ablations", nargs = "*", default = ["joint", "features", "range"],
                        choices = ["joint", "features", "range"])
    parser.add_argument("--groups", nargs = "*", default = ["snr"], choices = list(FEATURE_GROUPS),
                        help = "feature groups to drop one at a time (default : snr)")
    args = parser.parse_args()
    if "joint" in args.ablations:
        ablation_joint()
    if "features" in args.ablations:
        ablation_features(args.groups)
    if "range" in args.ablations:
        ablation_range()
