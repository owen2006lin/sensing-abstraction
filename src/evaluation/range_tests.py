from src.data.load import load_pos, group_split
from src.components.position import FullPositionModel
from src.components.range.range import RangeModel
from src.evaluation.error_metrics import (scene_ids, tile_frame, to_draws, w1_null, plot_w1_null, score_models,
                                          print_table, slice_metrics, plot_slices, plot_marginals)
import numpy as np
import polars as pl


MODEL_PATH = "models/position_model.joblib"
ERROR_COL = "range_error_m"

# Same group split as the notebook that trained position_model, so X_val is held out
tags, labels, features = load_pos()
[X, X_val, y, y_val, t, t_val] = group_split(tags, labels, features, return_tags = True)

pos_model = FullPositionModel.load(MODEL_PATH)
range_model = pos_model.range_model

y_train = y[ERROR_COL].to_numpy()
y_true = y_val[ERROR_COL].to_numpy()
scene = scene_ids(t_val)


# (1) Draws matrix from the range model : (n_points, n_draws)
# Regime probabilities only depend on the features, so predict once and tile, then sample every copy in one call

def sample_range(model : RangeModel, X, n_draws : int, rng):
    probs = model.predict_proba(X)
    samples = model.sample(tile_frame(X, n_draws), np.tile(probs, (n_draws, 1)), rng)
    return to_draws(samples["sampled_error"].to_numpy(), len(X), n_draws)


# (2) Baselines, same draws format
# Climatology : resample the training errors, ignores the features (the CRPS reference)
def climatology_draws(y_train, n_points : int, n_draws : int, rng):
    return rng.choice(y_train, size = (n_points, n_draws), replace = True)

# Gaussian : single N(mean, std) fit on the training errors, same error model as B1 (ConstantClassifier)
def gaussian_draws(y_train, n_points : int, n_draws : int, rng):
    return rng.normal(y_train.mean(), y_train.std(), size = (n_points, n_draws))


SLICE_FEATURES = ["bistatic_range_m", "nn_norm_range_sep"]

#--------------------------------------------------------------------------------------------------------------------#

# Example usage:

# ---- Setup : draws for the range model and baselines ----
N_DRAWS = 30
rng = np.random.default_rng(seed = 42)

models = {
    "climatology" : climatology_draws(y_train, len(y_true), N_DRAWS, rng),
    "gaussian" : gaussian_draws(y_train, len(y_true), N_DRAWS, rng),
    "range model" : sample_range(range_model, X_val, N_DRAWS, rng)
}

# ---- (1) - (5) Pooled W1, MC null p, CRPS + CRPSS vs climatology, scene bootstrap CIs ----
scores = score_models(y_true, scene, models, crps = True, ref = "climatology", n_boot = 1000)
print_table(scores, "range error (m)")

# ---- MC null W1 for the range model ----
observed, null, p_value = w1_null(y_true, models["range model"])
plot_w1_null(observed, null, p_value, "range error", "m")

plot_marginals(y_true, models, "range error marginal", "range error (m)")

# ---- (6) W1 / CRPS sliced by range and neighbor separation ----
for feature in SLICE_FEATURES:
    slices = slice_metrics(X_val, y_true, models, feature, n_bins = 8, crps = True)
    plot_slices(slices, feature, "range error")
