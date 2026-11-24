import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import contextlib
import numpy as np
import polars as pl
from src.data.load import load_pos, group_split
from src.components.position import FullPositionModel
from src.evaluation.range_tests import ERROR_COL, SLICE_FEATURES, sample_range, climatology_draws, gaussian_draws
from src.evaluation.error_metrics import (scene_ids, score_models, print_table, w1_null, plot_w1_null, plot_marginals,
                                          slice_metrics, plot_slices)

# Runs every range error test on the held out test set and saves tables, figures and a summary to results/range/
# Run from the repo root after scripts/train.py : python scripts/range_test.py
MODEL_PATH = "models/position_model.joblib"
OUT = Path("results/range")
FIGS = OUT / "figures"
FIGS.mkdir(parents = True, exist_ok = True)
N_DRAWS = 30


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

# the range model inside position_model is the one the full pipeline samples from
range_model = FullPositionModel.load(MODEL_PATH).range_model


#--------------------------------Draws : (n_points, n_draws) for every model-------------------------------#
# climatology : resampled training errors, gaussian : single N(mean, std), range model : regime classifier + error models
rng = np.random.default_rng(seed = 42)
models = {
    "climatology" : climatology_draws(y_train, len(y_true), N_DRAWS, rng),
    "gaussian" : gaussian_draws(y_train, len(y_true), N_DRAWS, rng),
    "range model" : sample_range(range_model, X_test, N_DRAWS, rng)
}


#--------------------------------(1) - (5) Pooled W1, MC null p, CRPS + CRPSS vs climatology, scene bootstrap CIs-------------------------------#
scores = score_models(y_true, scene, models, crps = True, ref = "climatology", n_boot = 1000)
scores.write_csv(OUT / "scores.csv")


#--------------------------------MC null W1 for the range model, and real vs model marginals-------------------------------#
observed, null, p_value = w1_null(y_true, models["range model"])
save_fig(plot_w1_null(observed, null, p_value, "range error", "m"), "w1_null_range_model")
save_fig(plot_marginals(y_true, models, "range error marginal", "range error (m)"), "marginals")


#--------------------------------(6) W1 / CRPS sliced by range and neighbor separation-------------------------------#
all_slices = []
for feature in SLICE_FEATURES:
    slices = slice_metrics(X_test, y_true, models, feature, n_bins = 8, crps = True)
    all_slices.append(slices.with_columns(pl.lit(feature).alias("feature")))
    for metric, fig in plot_slices(slices, feature, "range error").items():
        save_fig(fig, f"slice_{metric}_{feature}")
pl.concat(all_slices).write_csv(OUT / "slices.csv")


#--------------------------------Summary-------------------------------#
with pl.Config(tbl_rows = -1, tbl_cols = -1, tbl_width_chars = 200, fmt_str_lengths = 40), io.StringIO() as buf:
    with contextlib.redirect_stdout(buf):
        print(f"test set : {len(y_true)} targets, {N_DRAWS} draws per target")
        print_table(scores, "range error (m) : value [95% scene bootstrap CI]")
    summary = buf.getvalue()

(OUT / "summary.md").write_text(f"# Range test results\n\n```\n{summary}```\n")
print(summary)
