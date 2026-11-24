import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import contextlib
import numpy as np
import polars as pl
from src.data.load import load_pos, group_split
from src.components.position import FullPositionModel
from src.evaluation.position_classes import QuantileErrorModel, ERROR_COL
from src.evaluation.positional_tests import (SLICE_FEATURES, sample_position, climatology_draws, direct_draws,
                                             c2st_models, plot_c2st_null, plot_c2st_importance)
from src.evaluation.error_metrics import (scene_ids, score_models, print_table, w1_null, plot_w1_null, plot_marginals,
                                          slice_metrics, plot_slices)

# Runs every 3D position error test on the held out test set and saves tables, figures and a summary to results/position/
# Run from the repo root after scripts/train.py : python scripts/position_test.py
MODEL_PATH = "models/position_model.joblib"
BASELINE_PATH = "src/evaluation/models/quantile.joblib"
OUT = Path("results/position")
FIGS = OUT / "figures"
FIGS.mkdir(parents = True, exist_ok = True)
# n_draws >= 2 * n_null + 1 so the C2ST null pairs come from the same draws
N_DRAWS = 41
N_NULL = 20


def save_fig(fig, name):
    fig.write_html(FIGS / f"{name}.html", include_plotlyjs = "cdn")


#--------------------------------Setup : same two step split as scripts/train.py-------------------------------#
# outer split holds out the test set, inner split gives the early stopping val set the models were trained with
tags, labels, features = load_pos()
[X, X_test, y, y_test, t, t_test] = group_split(tags, labels, features, return_tags = True)
[X, X_val, y, y_val, t, t_val] = group_split(t, y, X, return_tags = True)
scene = scene_ids(t_test)
y_train = y[ERROR_COL].to_numpy()
y_true = y_test[ERROR_COL].to_numpy()

pos_model = FullPositionModel.load(MODEL_PATH)


#--------------------------------Baseline : direct quantile GBM, same train / early stopping val sets as position_model-------------------------------#
direct = QuantileErrorModel().fit(X, y, X_val, y_val)
direct.save(BASELINE_PATH)


#--------------------------------Draws : (n_points, n_draws) for every model-------------------------------#
# climatology : resampled training errors, direct quantile : features -> 3D error, position model : range + angle + geometry
rng = np.random.default_rng(seed = 42)
models = {
    "climatology" : climatology_draws(y_train, len(y_true), N_DRAWS, rng),
    "direct quantile" : direct_draws(direct, X_test, N_DRAWS, rng),
    "position model" : sample_position(pos_model, X_test, N_DRAWS, rng)
}


#--------------------------------(1) - (2) Pooled W1, MC null p, CRPS + CRPSS vs climatology, scene bootstrap CIs-------------------------------#
scores = score_models(y_true, scene, models, crps = True, ref = "climatology", n_boot = 1000)
scores.write_csv(OUT / "scores.csv")


#--------------------------------MC null W1 for the direct baseline and position model, and real vs model marginals-------------------------------#
for name in ["direct quantile", "position model"]:
    observed, null, p_value = w1_null(y_true, models[name])
    save_fig(plot_w1_null(observed, null, p_value, f"3D error {name}", "m"), f"w1_null_{name.replace(' ', '_')}")
save_fig(plot_marginals(y_true, models, "3D position error marginal", "3D error (m)"), "marginals")


#--------------------------------W1 / CRPS sliced by geometry and neighbor separation-------------------------------#
all_slices = []
for feature in SLICE_FEATURES:
    slices = slice_metrics(X_test, y_true, models, feature, n_bins = 8, crps = True)
    all_slices.append(slices.with_columns(pl.lit(feature).alias("feature")))
    for metric, fig in plot_slices(slices, feature, "3D error").items():
        save_fig(fig, f"slice_{metric}_{feature}")
pl.concat(all_slices).write_csv(OUT / "slices.csv")


#--------------------------------(3) C2ST : real vs synthetic discriminator, null from synthetic vs synthetic-------------------------------#
c2st_table, c2st_results = c2st_models(X_test, y_true, scene, models, n_null = N_NULL)
c2st_table.write_csv(OUT / "c2st.csv")
for name, (c2st, null_aucs, p_value) in c2st_results.items():
    save_fig(plot_c2st_null(c2st, null_aucs, p_value, name), f"c2st_null_{name.replace(' ', '_')}")
    save_fig(plot_c2st_importance(c2st, name), f"c2st_importance_{name.replace(' ', '_')}")


#--------------------------------Summary-------------------------------#
with pl.Config(tbl_rows = -1, tbl_cols = -1, tbl_width_chars = 200, fmt_str_lengths = 40), io.StringIO() as buf:
    with contextlib.redirect_stdout(buf):
        print(f"test set : {len(y_true)} targets, {N_DRAWS} draws per target")
        print_table(scores, "3D position error (m) : value [95% scene bootstrap CI]")
        print("\nC2ST")
        print(c2st_table)
    summary = buf.getvalue()

(OUT / "summary.md").write_text(f"# 3D position test results\n\n```\n{summary}```\n")
print(summary)
