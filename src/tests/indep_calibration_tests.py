import lightgbm as lgb
from src.load import *
from sklearn.model_selection import train_test_split
from typing import cast
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


# (1) Confusion matrix, computes best f1 threshold, and plot additional thresholds if needed
from sklearn.metrics import precision_recall_curve, matthews_corrcoef
from sklearn.metrics import confusion_matrix

def best_f1(preds, labels):
    prec, rec, thr = precision_recall_curve(labels, preds)
    f1 = 2*prec*rec / (prec + rec + 1e-12)

    i = np.argmax(f1[:-1])
    best_threshold = thr[i]
    b_f1 = f1[i]

    return best_threshold, b_f1

def plot_confusion(preds, labels, thresholds : list[float] = [0.3, 0.5]):

    thresholds = thresholds.copy()
    b_threshold, b_f1 = best_f1(preds, labels)
    subplot_titles=[f"Threshold: {t:.2f}" for t in thresholds]
    subplot_titles.append(f"Best f1: {b_f1:.2f}, Threshold: {b_threshold:.2f}")
    thresholds.append(b_threshold)

    n = len(thresholds)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig = make_subplots(rows = rows, cols = cols, subplot_titles=subplot_titles,horizontal_spacing = 0.08)
    for i, thr in enumerate(thresholds):
        r, c = divmod(i, cols)
        temp = (preds >= thr).astype(int)
        cm = confusion_matrix(labels, temp)

        fig.add_trace(
            go.Heatmap(z = cm,
                x = ["Model Predicts Missed", "Model Predicts Detected"],
                y = ["ISAC Missed", "ISAC Detected"],
                text = cm, texttemplate="%{text}", coloraxis = "coloraxis"),
            row = r+1, col = c+1
        )
    fig.update_yaxes(autorange="reversed")
    fig.show()

#--------------------------------------------------------------------------------------------------------------------#

# Example usage:
# plot_confusion(preds, y_val)
# plot_confusion(preds_gss, y_val_gss)

#--------------------------------------------------------------------------------------------------------------------#


# (2) Reliability Diagram : Note that plotting raw scores may be misleading, preferred to use logits
from scipy.special import logit, expit


def chunk(n_bins : int):
    zipped = zip(preds, y_val)
    s = sorted(zipped)

    chunks = [chunk.tolist() for chunk in np.array_split(s, n_bins)]

    p_hat=[]
    freq = []
    for chunk in chunks:
        sums = column_sums = list(map(sum, zip(*chunk)))
        mean_p = sums[0] / len(chunk)
        prob = sums[1] / len(chunk)

        p_hat.append(mean_p)
        freq.append(prob)
    return p_hat, freq

# Plotting with raw probabilities, but many points are clustered at 1.0, so may not be that informative
def plot_reliability(n_bins : int):
    p_hat, freq = chunk(n_bins)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash", color="gray", width=1.5)))
    fig.add_trace(go.Scatter(x=p_hat, y=freq, mode="markers+lines", name="model",
                             marker=dict(size=8)))
    fig.update_layout(
        xaxis=dict(title="mean predicted probability", range=[0, 1], constrain="domain"),
        yaxis=dict(title="observed frequency", range=[0, 1],
                   scaleanchor="x", scaleratio=1),
        width=520, height=520, template="simple_white",
    )
    fig.show()

# Instead, we should plot the logits using a log scale, so the groups aren't too close together visually
TICKS = np.array([0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999])

def plot_reliability_logits(n_bins : int, ticks = TICKS):
    p_hat, freq = chunk(n_bins)

    # Clip between epsilon and 1-epsilon to prevent instability with logits operation
    epsilon = 0.5/(len(preds)/n_bins)
    p_hat_clipped = np.clip(p_hat, epsilon, 1 - epsilon)
    freq_clipped = np.clip(freq, epsilon, 1 - epsilon)

    ticks_logit = logit(ticks)
    min_tick, max_tick = ticks_logit.min(), ticks_logit.max()


    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[min_tick, max_tick], y=[min_tick, max_tick], mode="lines", name="perfect",
                                 line=dict(dash="dash", color="gray", width=1.5))
    )
    fig.add_trace(
        go.Scatter(x = logit(p_hat_clipped), y = logit(freq_clipped),mode="markers", name="model",
                                 marker=dict(size=8), customdata=np.stack([p_hat, freq], -1),
                                 hovertemplate="p̂ %{customdata[0]:.4f}<br>obs %{customdata[1]:.4f}")
    )
    axis = dict(tickvals=logit(ticks), ticktext=[str(t) for t in ticks], range=[min_tick, max_tick])
    fig.update_layout(
        xaxis=dict(title="mean predicted probability", **axis),
        yaxis=dict(title="observed frequency", scaleanchor="x", scaleratio=1, **axis),
        width=560, height=560, template="simple_white")
    fig.show()




#--------------------------------------------------------------------------------------------------------------------#

# Example usage:
plot_reliability(20)
plot_reliability_logits(20)

#--------------------------------------------------------------------------------------------------------------------#
