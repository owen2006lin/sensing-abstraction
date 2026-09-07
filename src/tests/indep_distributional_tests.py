import lightgbm as lgb
import numpy as np
from src.load import *
from src.features import *
from typing import cast
import plotly.graph_objects as go


MODEL_PATH = "models/indep_model.txt"
MODEL_GSS_PATH = "models/indep_model_gss.txt"


_ , labels, features = load_indep()
[X_train, X_val, y_train, y_val] =  train_test_split(features, labels, test_size=0.2, random_state=42)
model = lgb.Booster(model_file = MODEL_PATH)
preds = cast(np.ndarray, model.predict(X_val))
y_val = y_val.to_numpy().ravel()


tags, labels, features = load_indep()
groups = tags.select(pl.struct(["scenario_id", "drop_id"]).hash()).to_series().to_numpy()
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(features, labels, groups=groups))
X_val_gss, y_val_gss = features[val_idx], labels[val_idx]

model_gss = lgb.Booster(model_file = MODEL_GSS_PATH)
preds_gss = model_gss.predict(X_val_gss)
y_val_gss = y_val_gss.to_numpy().ravel()



# (1) Kolmogorov-Smirnov test
from scipy.stats import ks_2samp


def ks_test(preds, labels):
    combined = list(zip(preds, labels))

    scores_pos = [p for (p,l) in combined if l == 1]
    scores_neg = [p for (p,l) in combined if l == 0]

    ks_stat, p_value = ks_2samp(scores_pos, scores_neg)
    return ks_stat, p_value

# ks_stat, p_value = ks_test(preds, y_val)
# print(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4g}")

# (2) Anderson-Darling
from scipy.stats import anderson_ksamp, PermutationMethod

def ad_test(preds, labels):
    combined = list(zip(preds, labels))
    scores_pos = [p for (p,l) in combined if l == 1]
    scores_neg = [p for (p,l) in combined if l == 0]

    res = anderson_ksamp(
        [scores_neg, scores_pos], method = PermutationMethod(n_resamples = 9999)
    )

    return res.statistic, res.pvalue

# print(ad_test(preds, y_val))

# (3) Chi squared test
def chunk(data, preds, labels, feature, n_bins : int):
    col = data.select(feature).to_numpy().ravel()
    zipped = zip(col, preds, labels)
    s = sorted(zipped, key = lambda x : x[0])

    chunks = [chunk.tolist() for chunk in np.array_split(s, n_bins)]

    ret = []
    for chunk in chunks:
        sums = column_sums = list(map(sum, zip(*chunk)))
        observed = sums[2]
        expected = sums[1]
        var = 0
        for (_, p, _) in chunk:
            var += p * (1-p)

        z_bin = (observed - expected) / np.sqrt(var)

        ret.append((observed, expected, z_bin))
    return ret

def chi_squared(data, preds, labels, feature, n_bins):
    points = chunk(data, preds, labels, feature, n_bins)
    chi_sum = 0
    Z_scores = []
    for point in points:
        _, _, z_bin = point
        chi_sum += (z_bin)**2
        Z_scores.append(z_bin)

    return chi_sum, Z_scores

PRIMARY_FEATURES = [
                        IndepFeature.RSS_MAX, 
                        IndepFeature.RSS_TOTAL_DB, 
                        IndepFeature.ABS_BISTATIC_DOPPLER_HZ,
                        IndepFeature.NN_NORM_RANGE_SEP,
                        IndepFeature.AZ_OFF_BORESIGHT
                    ]

LIM, WIDTH = 4.0, 0.5          # same axis + same bins for all features

