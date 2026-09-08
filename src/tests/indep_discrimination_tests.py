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

ks_stat, p_value = ks_test(preds, y_val)
print(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4g}")

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

print(ad_test(preds, y_val))




