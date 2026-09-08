import lightgbm as lgb
from src.load import *
from src.features import *
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


def chunk(preds, labels, n_bins : int):
    zipped = zip(preds, labels)
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


def plot_reliability(preds, labels, n_bins : int):
    p_hat, freq = chunk(preds, labels, n_bins)

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

# (3) Expected calibration error



def ece_adaptive(preds, labels, n_bins):
    n = len(preds)
    conf, acc = chunk(preds, labels, n_bins)
    ece = 0
    chunks = [chunk.tolist() for chunk in np.array_split(preds, n_bins)]

    for i in range(len(conf)):
        diff = np.abs(conf[i] - acc[i])
        scale = len(chunks[i])/n
        ece += scale * diff

    return ece


def ece(preds, labels, n_bins):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    n = len(preds)

    order = np.argsort(preds)
    preds_sorted = preds[order]
    labels_sorted = labels[order]

    bin_edges = np.round(np.linspace(0, n, n_bins + 1)).astype(int)

    total_error = 0.0
    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        if start == end:
            continue

        chunk_preds = preds_sorted[start:end]
        chunk_labels = labels_sorted[start:end]

        avg_confidence = chunk_preds.mean()
        avg_accuracy = chunk_labels.mean()
        bin_weight = (end - start) / n

        total_error += bin_weight * abs(avg_confidence - avg_accuracy)

    return total_error




# (4) Hosmer - Lemeshow
from scipy.stats import chi2

def hosmer_lemeshow(preds, labels, n_bins):
    zipped = zip(preds, labels)
    s = sorted(zipped)
    chunks = [chunk.tolist() for chunk in np.array_split(s, n_bins)]

    hl_val = 0

    for chunk in chunks:
        n = len(chunk)
        sums = list(map(sum, zip(*chunk)))
        expected = sums[0]
        observed = sums[1]

        term = (observed - expected)**2 / (expected*(1-(expected/n)))

        hl_val += term

    dof = n_bins - 2
    p_value = chi2.sf(hl_val, dof)
    return hl_val, p_value


# (5) Spieghalter Z-test

def spieghalter(preds, labels) -> int:
    zipped = zip(preds, labels)

    top = 0
    bottom = 0
    for (pred, label) in zipped:
        top += (label - pred)*(1 - (2*pred))
        bottom += ((1 - (2*pred))**2) * pred * (1 - pred)

    bottom = np.sqrt(bottom)
    return (top/bottom)



# (6) Brier Score + Murphy's Deomposition

def brier(preds, labels):
    n = len(preds)
    zipped = zip(preds, labels)
    score = 0

    for (pred, label) in zipped:
        score += (pred - label)**2

    return score / n

def murphy_decomp(preds, labels, n_bins):
    b_score = brier(preds, labels)
    N = len(preds)

    zipped = zip(preds, labels)
    s = sorted(zipped)
    o = sum(b for a, b in s) / N

    rel = 0
    res = 0
    unc = o * (1 - o)




    chunks = [chunk.tolist() for chunk in np.array_split(s, n_bins)]
    for chunk in chunks:
        n = len(chunk)
        sums = list(map(sum, zip(*chunk)))

        mean_p = sums[0] / len(chunk)
        prob = sums[1] / len(chunk)
        val = n * (mean_p - prob)**2
        rel += val

        res += (prob - o)**2 * n

    rel = rel / N
    res = res / N

    return {
        "brier_score": b_score,
        "reliability": rel,
        "resolution": res,
        "uncertainty": unc,
        "reconstructed_brier": rel - res + unc,  # should ≈ brier_score
    }




# (7) Probability curve vs covariates

# TreeSHAP for variance decomposition
import shap
import matplotlib.pyplot as plt
from scipy.stats import beta
def shap_explain(model : lgb.Booster, data : np.ndarray):
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    data = X_val.to_pandas()

    shap.summary_plot(shap_values, data, show = False)
    plt.savefig("figures/shap_summary.png")
    plt.close()



# From the figure, we can see primary features are:
PRIMARY_FEATURES = [
                        IndepFeature.RSS_MAX, 
                        IndepFeature.RSS_TOTAL_DB, 
                        IndepFeature.ABS_BISTATIC_DOPPLER_HZ,
                        IndepFeature.NN_NORM_RANGE_SEP,
                        IndepFeature.AZ_OFF_BORESIGHT
                    ]

def p_lower(n, x, alpha = 0.05):
    if x == 0:
        return 0
    else:
        return beta.ppf(alpha/2, x, n - x + 1)

def p_upper(n, x, alpha = 0.05):
    if x == n:
        return 1 
    else:
        return beta.ppf(1 - alpha/2, x + 1, n - x)


    
def clopper_pearson(model, data, labels, feature, n_bins):
    preds = model.predict(data)
    preds = pl.DataFrame({"prediction": preds})
    labels = pl.DataFrame({"detected": labels})


    col = data.select(feature)
    combined = pl.concat([col, preds, labels], how = "horizontal")
    s = combined.sort(feature)

    chunks = [chunk.tolist() for chunk in np.array_split(s, n_bins)]

    points = []

    for chunk in chunks:
        n = len(chunk)
        sums = list(map(sum, zip(*chunk)))

        detections = sums[2]
        p_hat = detections / n
        lower = p_lower(n, detections)
        upper = p_upper(n, detections)

        f_mid = sums[0] / n
        mean_pred = sums[1] / n
        points.append((f_mid, p_hat, lower, upper, mean_pred))

    return points

def plot_clopper_pearson(model, data, labels, features, n_bins):
    preds = model.predict(data)

    for feature in features:
        points = clopper_pearson(model, data, labels, feature, n_bins)
        f_mid, p_hat, lower, upper, mean_pred = zip(*points)  # each becomes a tuple

        array = [u - p for u, p in zip(upper, p_hat)]
        arrayminus = [p - l for p, l in zip(p_hat, lower)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = f_mid,
            y = mean_pred,
            mode='lines+markers',
            name = 'Model prediction',
            line = dict(color = 'royalblue')
        )
        )
        fig.add_trace(go.Scatter(
            x=f_mid,
            y=p_hat,
            mode='markers',
            name=f'{feature} Empirical Pd (95% CI)',
            marker=dict(size=8, color='firebrick'),
            error_y=dict(
                type='data',
                symmetric=False,
                array=array,
                arrayminus=arrayminus,
                visible=True,
                thickness=2,
                width=6
            )
        ))

        fig.write_html(f"figures/clopper_pearson/{feature}_plot.html")

# plot_clopper_pearson(model, X_val, y_val, PRIMARY_FEATURES, 20)
        
# (8) Chi squared test
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


def plot_z_scores(data, preds, labels, features, n_bins):
    for feature in features:
        chi_sum, Z_scores = chi_squared(data, preds, labels, feature, n_bins)
        fig = go.Figure(
            data = [
                go.Bar(y = Z_scores)
            ]
        )

        fig.update_layout(
            title = f"{feature.value} Chi Squared : {chi_sum} ",
            yaxis_title = "Z-score",
            bargap = 0.05
        )

        fig.write_html(f"figures/chi_squared/{feature.value}_zscore.html")

#plot_z_scores(X_val, preds, y_val, PRIMARY_FEATURES, 8)

