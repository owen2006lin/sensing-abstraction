import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import contextlib
import polars as pl
from src.data.load import load_class, group_split
from src.components.detection import FullClassifier
from src.evaluation.detection_classes import ConstantClassifier, MDNClassifier, PairMDNClassifier, MDN_PARAMS, PAIR_MDN_PARAMS
from src.evaluation.detection_tests import (baseline_labels, to_branch_preds, constant_preds, score_models, print_tables,
                                            mc_null_ece, plot_mc_null, SLICE_FEATURES, slice_metrics, plot_slices,
                                            c2st_models, plot_c2st_null, plot_c2st_importance)

# Runs every detection test on the held out test set and saves tables, figures and a summary to results/detection/
# Run from the repo root after scripts/train.py : python scripts/detection_test.py
MODEL_PATH = "models/full_classifier.joblib"
BASELINE_DIR = "src/evaluation/models/"
OUT = Path("results/detection")
FIGS = OUT / "figures"
FIGS.mkdir(parents = True, exist_ok = True)
N_BINS = 15


def save_fig(fig, name):
    fig.write_html(FIGS / f"{name}.html", include_plotlyjs = "cdn")


#--------------------------------Setup : same two step split as scripts/train.py-------------------------------#
# outer split holds out the test set, inner split gives the early stopping val set the models were trained with
tags, labels, features = load_class()
[X, X_test, y, y_test, t, t_test] = group_split(tags, labels, features, return_tags = True)
[X, X_val, y, y_val, t, t_val] = group_split(t, y, X, return_tags = True)

train_df = pl.concat([X, y, t], how = "horizontal")
val_df = pl.concat([X_val, y_val, t_val], how = "horizontal")
inf_df = pl.concat([X_test, t_test], how = "horizontal")
y_true = y_test.to_numpy().ravel()

full_classifier = FullClassifier.load(MODEL_PATH)
indep_preds, pair_preds = full_classifier.predict_proba(inf_df)
n_indep, n_pairs = len(indep_preds), len(pair_preds)


#--------------------------------Baselines, all in (indep_preds, pair_preds) format-------------------------------#
# constant classifier : one detection rate for every point, pairs treated as independent
constant = ConstantClassifier().fit(X, baseline_labels(y, t))
# MLP : same indep / mutual split as full_classifier, with sklearn MLPs instead of LightGBM
mlp = FullClassifier(models = [MDNClassifier(), PairMDNClassifier()], params = [MDN_PARAMS, PAIR_MDN_PARAMS],
                     callbacks = [None, None]).fit(train_df, val_df)
constant.save(BASELINE_DIR + "constant.joblib")
mlp.save(BASELINE_DIR + "mlp.joblib")

models = {
    "constant rate" : constant_preds(train_df, indep_preds, pair_preds),
    "constant classifier" : to_branch_preds(full_classifier, inf_df, constant.predict_proba(inf_df), n_indep, n_pairs),
    "MLP" : mlp.predict_proba(inf_df),
    "full classifier" : (indep_preds, pair_preds)
}


#--------------------------------(5), (6) Brier + Murphy decomposition, AUROC, scene bootstrap CIs-------------------------------#
scores = score_models(full_classifier, inf_df, y_true, models, n_bins = N_BINS, n_boot = 1000)
scores.write_csv(OUT / "scores.csv")


#--------------------------------(3) MC null ECE-------------------------------#
# full_classifier is only used for its generic samplers + reassemble(), so it samples every model's joint
mc_rows = []
for name, preds in models.items():
    observed, null_eces, p_value = mc_null_ece(full_classifier, inf_df, y_true, *preds, n_bins = N_BINS)
    mc_rows.append({"model" : name, "ece" : observed, "null_mean" : null_eces.mean(), "p_value" : p_value})
    save_fig(plot_mc_null(observed, null_eces, p_value, n_bins = N_BINS), f"mc_null_ece_{name.replace(' ', '_')}")
mc_table = pl.DataFrame(mc_rows)
mc_table.write_csv(OUT / "mc_null_ece.csv")


#--------------------------------(7) AUROC / Brier sliced by range and angle-------------------------------#
for feature in SLICE_FEATURES:
    slices = slice_metrics(full_classifier, inf_df, y_true, models, feature, n_bins = 8)
    slices.write_csv(OUT / f"slices_{feature}.csv")
    for metric, fig in plot_slices(slices, feature).items():
        save_fig(fig, f"slice_{metric}_{feature}")


#--------------------------------(9) C2ST : real vs synthetic, null from synthetic vs synthetic-------------------------------#
c2st_table, c2st_results = c2st_models(full_classifier, inf_df, y_true, models, n_null = 20)
c2st_table.write_csv(OUT / "c2st.csv")
for name, (c2st, null_aucs, p_value) in c2st_results.items():
    save_fig(plot_c2st_null(c2st, null_aucs, p_value, name), f"c2st_null_{name.replace(' ', '_')}")
    save_fig(plot_c2st_importance(c2st, name), f"c2st_importance_{name.replace(' ', '_')}")


#--------------------------------Summary-------------------------------#
with pl.Config(tbl_rows = -1, tbl_cols = -1, tbl_width_chars = 200, fmt_str_lengths = 40), io.StringIO() as buf:
    with contextlib.redirect_stdout(buf):
        print(f"test set : {len(y_true)} targets, {n_indep} non mutual, {n_pairs} mutual pairs\n")
        print("Brier / Murphy decomposition / AUROC : value [95% scene bootstrap CI]")
        print_tables(scores)
        print("\nMC null ECE")
        print(mc_table)
        print("\nC2ST")
        print(c2st_table)
    summary = buf.getvalue()

(OUT / "summary.md").write_text(f"# Detection test results\n\n```\n{summary}```\n")
print(summary)
