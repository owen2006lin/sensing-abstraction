from src.components.angular.quantization import AngleModel
from src.evaluation.error_metrics import tile_frame, to_draws
import numpy as np
import polars as pl

# Draw generators for the angle error tests, scored with the shared metrics in src.evaluation.error_metrics
# scripts/angle_test.py runs every test and saves the results


# model output column -> ISAC label column, az and el are tested separately
AXES = {
    "az" : ("az_err", "rx_azimuth_error_deg"),
    "el" : ("el_err", "rx_elevation_error_deg")
}


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


# W1 is sliced by these covariates to check p(error | feature), not just the pooled marginal
SLICE_FEATURES = ["az_off_boresight", "rx_elevation_deg", "nn_norm_range_sep"]
