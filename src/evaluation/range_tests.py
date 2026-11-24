from src.components.range.range import RangeModel
from src.evaluation.error_metrics import tile_frame, to_draws
import numpy as np

# Draw generators for the range error tests, scored with the shared metrics in src.evaluation.error_metrics
# scripts/range_test.py runs every test and saves the results


# ISAC label column the range model is compared against
ERROR_COL = "range_error_m"


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

# Gaussian : single N(mean, std) fit on the training errors, same error model as the constant classifier (ConstantClassifier)
def gaussian_draws(y_train, n_points : int, n_draws : int, rng):
    return rng.normal(y_train.mean(), y_train.std(), size = (n_points, n_draws))


# W1 / CRPS are sliced by these covariates to check p(error | feature), not just the pooled marginal
SLICE_FEATURES = ["bistatic_range_m", "nn_norm_range_sep"]
