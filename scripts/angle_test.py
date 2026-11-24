import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import contextlib
import numpy as np
import polars as pl
from src.data.load import load_pos, group_split
from src.components.position import FullPositionModel
from src.evaluation.angle_tests import AXES, SLICE_FEATURES, sample_angles, no_slip_draws, climatology_draws
from src.evaluation.error_metrics import (scene_ids, score_models, print_table, w1_null, plot_w1_null, plot_marginals,
                                          slice_metrics, plot_slices)

# Runs every angle error test (az and el separately) on the held out test set and saves tables, figures and a summary
# to results/angle/. Run from the repo root after scripts/train.py : python scripts/angle_test.py
MODEL_PATH = "models/position_model.joblib"
OUT = Path("results/angle")
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

angle_model = FullPositionModel.load(MODEL_PATH).angle_model


#--------------------------------Draws : (n_points, n_draws) per axis for every model-------------------------------#
# climatology : resampled training errors, no slip : grid snapping only, angle model : grid snapping + slips
rng = np.random.default_rng(seed = 42)
draws = {
    "climatology" : climatology_draws(y, len(X_test), N_DRAWS, rng),
    "no slip" : no_slip_draws(angle_model, X_test, N_DRAWS, rng),
    "angle model" : sample_angles(angle_model, X_test, N_DRAWS, rng)
}


#--------------------------------Tests, az and el separately-------------------------------#
all_scores, all_slices, tables = [], [], io.StringIO()
for axis, (_, label_col) in AXES.items():
    y_true = y_test[label_col].to_numpy()
    models = {name : d[axis] for name, d in draws.items()}

    # (1) - (5) Pooled W1, MC null p, scene bootstrap CIs
    scores = score_models(y_true, scene, models, n_boot = 1000)
    all_scores.append(scores.with_columns(pl.lit(axis).alias("axis")))
    with pl.Config(tbl_rows = -1, tbl_cols = -1, tbl_width_chars = 200, fmt_str_lengths = 40), contextlib.redirect_stdout(tables):
        print_table(scores, f"{axis} error (deg) : value [95% scene bootstrap CI]")

    # MC null W1 for the angle model, and real vs model marginals
    observed, null, p_value = w1_null(y_true, models["angle model"])
    save_fig(plot_w1_null(observed, null, p_value, f"{axis} error", "deg"), f"w1_null_{axis}")
    save_fig(plot_marginals(y_true, models, f"{axis} error marginal", f"{axis} error (deg)"), f"marginals_{axis}")

    # (6) W1 sliced by geometry and neighbor separation
    for feature in SLICE_FEATURES:
        slices = slice_metrics(X_test, y_true, models, feature, n_bins = 8)
        all_slices.append(slices.with_columns(pl.lit(axis).alias("axis"), pl.lit(feature).alias("feature")))
        for metric, fig in plot_slices(slices, feature, f"{axis} error").items():
            save_fig(fig, f"slice_{metric}_{axis}_{feature}")

pl.concat(all_scores).write_csv(OUT / "scores.csv")
pl.concat(all_slices).write_csv(OUT / "slices.csv")


#--------------------------------Summary-------------------------------#
summary = f"test set : {len(X_test)} targets, {N_DRAWS} draws per target\n" + tables.getvalue()
(OUT / "summary.md").write_text(f"# Angle test results\n\n```\n{summary}```\n")
print(summary)
