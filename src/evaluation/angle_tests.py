from src.data.load import load_pos, group_split
from src.components.position import FullPositionModel
from src.components.angular.quantization import AngleModel
from src.evaluation.error_metrics import (scene_ids, tile_frame, to_draws, w1_null, plot_w1_null, score_models,
                                          print_table, slice_metrics, plot_slices, plot_marginals)
import numpy as np
import polars as pl


MODEL_PATH = "models/position_model.joblib"
# model output column -> ISAC label column, az and el are tested separately
AXES = {
    "az" : ("az_err", "rx_azimuth_error_deg"),
    "el" : ("el_err", "rx_elevation_error_deg")
}

# Same group split as the notebook that trained position_model, so X_val is held out
tags, labels, features = load_pos()
[X, X_val, y, y_val, t, t_val] = group_split(tags, labels, features, return_tags = True)

pos_model = FullPositionModel.load(MODEL_PATH)
angle_model = pos_model.angle_model
scene = scene_ids(t_val)


# (1) Draws matrices from the angle model : {axis : (n_points, n_draws)}
# Slip probabilities only depend on the features, so predict once and tile, then sample every copy in one call

def sample_angles(model : AngleModel, X, n_draws : int, rng, p_u = None, p_v = None):
    if p_u is None or p_v is None:
        p_u, p_v = model.predict_proba(X)
    samples = model.sample(tile_frame(X, n_draws), np.tile(p_u, n_draws), np.tile(p_v, n_draws), rng)
    return {axis : to_draws(samples[out_col].to_numpy(), len(X), n_draws) for axis, (out_col, _) in AXES.items()}


# (2) Baselines, same draws format
# Quantization only : the same grid model with every slip forced to 0, so the gap to the full model is what slips add
def no_slip_draws(model : AngleModel, X, n_draws : int, rng):
    zeros = np.zeros(len(X))
    return sample_angles(model, X, n_draws, rng, p_u = zeros, p_v = zeros)

# Climatology : resample the training errors per axis, ignores the features
def climatology_draws(y_train : pl.DataFrame, n_points : int, n_draws : int, rng):
    return {axis : rng.choice(y_train[label_col].to_numpy(), size = (n_points, n_draws), replace = True)
            for axis, (_, label_col) in AXES.items()}


SLICE_FEATURES = ["az_off_boresight", "rx_elevation_deg", "nn_norm_range_sep"]

#--------------------------------------------------------------------------------------------------------------------#

# Example usage:

# ---- Setup : draws for the angle model and baselines ----
N_DRAWS = 30
rng = np.random.default_rng(seed = 42)

draws = {
    "climatology" : climatology_draws(y, len(X_val), N_DRAWS, rng),
    "no slip" : no_slip_draws(angle_model, X_val, N_DRAWS, rng),
    "angle model" : sample_angles(angle_model, X_val, N_DRAWS, rng)
}

for axis, (_, label_col) in AXES.items():
    y_true = y_val[label_col].to_numpy()
    models = {name : d[axis] for name, d in draws.items()}

    # ---- (1) - (5) Pooled W1, MC null p, scene bootstrap CIs ----
    scores = score_models(y_true, scene, models, n_boot = 1000)
    print_table(scores, f"{axis} error (deg)")

    # ---- MC null W1 for the angle model ----
    observed, null, p_value = w1_null(y_true, models["angle model"])
    plot_w1_null(observed, null, p_value, f"{axis} error", "deg")

    plot_marginals(y_true, models, f"{axis} error marginal", f"{axis} error (deg)")

    # ---- (6) W1 sliced by geometry and neighbor separation ----
    for feature in SLICE_FEATURES:
        slices = slice_metrics(X_val, y_true, models, feature, n_bins = 8)
        plot_slices(slices, feature, f"{axis} error")
