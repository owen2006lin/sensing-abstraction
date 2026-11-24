from src.components.position import FullPositionModel
from src.evaluation.position_classes import QuantileErrorModel
from src.evaluation.c2st_classes import C2ST
from src.evaluation.error_metrics import tile_frame, to_draws
from src.data.features import PosFeature
import numpy as np
import polars as pl
import plotly.graph_objects as go

# Draw generators, baselines and the C2ST for the 3D position error tests, scored with the shared metrics in
# src.evaluation.error_metrics. scripts/position_test.py runs every test and saves the results
# plot_* functions return the plotly figure (call .show() or .write_html() on it)


# FullPositionModel output column for the exact 3D error, compared against ERROR_COL (src.evaluation.position_classes)
OUT_COL = "err_3d"


# (1) Draws matrix from the full position model : (n_points, n_draws)
# FullPositionModel.sample predicts range / slip probabilities internally, so tile the frame and sample every copy in one call

def sample_position(model : FullPositionModel, X, n_draws : int, rng):
    samples = model.sample(tile_frame(X, n_draws), rng)
    return to_draws(samples[OUT_COL].to_numpy(), len(X), n_draws)


# (2) Baselines, same draws format
# Climatology : resample the training errors, ignores the features (the CRPS reference)
def climatology_draws(y_train, n_points : int, n_draws : int, rng):
    return rng.choice(y_train, size = (n_points, n_draws), replace = True)

# Direct quantile : quantile GBM (QuantileErrorModel) straight from features to 3D error. Quantiles only depend on the features, so predict once and tile
def direct_draws(model : QuantileErrorModel, X, n_draws : int, rng):
    quantiles = model.predict_proba(X)
    samples = model.sample(np.tile(quantiles, (n_draws, 1)), rng)
    return to_draws(samples[OUT_COL].to_numpy(), len(X), n_draws)


# W1 / CRPS are sliced by these covariates to check p(error | feature), not just the pooled marginal
SLICE_FEATURES = ["bistatic_range_m", "rx_elevation_deg", "az_off_boresight", "nn_norm_range_sep"]

#--------------------------------------------------------------------------------------------------------------------#

# (3) C2ST : classifier two sample test
# Real rows = (features, ISAC 3D error), synthetic rows = (same features, 3D error from one column of the model's draws)
# Pooled W1 only sees the marginal, the discriminator sees the features too, so it tests p(error | features)
# Null : the same C2ST on two other independent draw columns, AUC under "model is the truth"
# Draw columns are independent, so column 0 is the synthetic set and columns (1, 2), (3, 4), ... are null pairs

C2ST_FEATURES = [f.value for f in PosFeature]

def c2st_frame(X, err, feature_cols = C2ST_FEATURES):
    return X.select(feature_cols).with_columns(pl.Series(OUT_COL, np.asarray(err, dtype = float)))


def run_c2st(X, real, draws, groups, n_null : int = 20, n_folds : int = 5):
    draws = np.asarray(draws)
    if draws.shape[1] < 2 * n_null + 1:
        raise ValueError(f"need at least {2 * n_null + 1} draws for {n_null} null sims, got {draws.shape[1]}")

    c2st = C2ST(n_folds = n_folds).fit(c2st_frame(X, real), c2st_frame(X, draws[:, 0]), groups)

    null_aucs = []
    for k in range(n_null):
        synth_a = c2st_frame(X, draws[:, 2 * k + 1])
        synth_b = c2st_frame(X, draws[:, 2 * k + 2])
        null_aucs.append(C2ST(n_folds = n_folds).fit(synth_a, synth_b, groups).auc)

    null_aucs = np.array(null_aucs)
    # +1 so p is never exactly 0 with a finite number of sims
    p_value = (1 + np.sum(null_aucs >= c2st.auc)) / (1 + n_null)
    return c2st, null_aucs, p_value


# models : {name : draws}, same format as score_models
def c2st_models(X, real, groups, models : dict, n_null : int = 20, n_folds : int = 5):
    rows = []
    results = {}
    for name, draws in models.items():
        c2st, null_aucs, p_value = run_c2st(X, real, draws, groups, n_null, n_folds)
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
        title = f"C2ST 3D error {name} ({len(null_aucs)} null sims) : p = {p_value:.4f}",
        xaxis_title = "out of fold AUC",
        yaxis_title = "count",
        template = "simple_white"
    )
    return fig

# Which features the discriminator leans on, eg err_3d x bistatic_range_m points at a range dependent miss
def plot_c2st_importance(c2st : C2ST, name : str, top_k : int = 15):
    imp = c2st.importance.head(top_k)
    fig = go.Figure(data = [go.Bar(x = imp["gain"], y = imp["feature"], orientation = "h")])
    fig.update_layout(
        title = f"C2ST 3D error {name} feature importance (gain)",
        xaxis_title = "gain",
        yaxis = dict(autorange = "reversed"),
        template = "simple_white"
    )
    return fig

#--------------------------------------------------------------------------------------------------------------------#
