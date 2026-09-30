import lightgbm as lgb
from src.load import *
from src.features import *
from sklearn.model_selection import train_test_split
from typing import cast
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODEL_PATH = "models/pair_model.txt"

tags, labels, features = load_pairs()
[X_train, X_val, y_train, y_val] =  train_test_split(features, labels, test_size=0.2, random_state=42)
model = lgb.Booster(model_file = MODEL_PATH)
preds = cast(np.ndarray, model.predict(features))

# (1) Distribution Histograms
def distribution_histograms(preds, labels, combined_mode : bool):
    label_counts = labels.select(pl.col("score").value_counts(sort=True))
    counts = label_counts.select(pl.col("score").struct.field("count"))
    n_3 = counts.item(0,0)
    n_2 = counts.item(1,0)
    n_1 = counts.item(2,0)
    n_0 = counts.item(3,0)

    l_vals = [n_0, n_1, n_2, n_3]

    m_vals = np.sum(preds, axis = 0)

    x_labels = ["neither_detected", "only b detected", "only a detected", "both detected"]

    if combined_mode:    
        m_vals = [m_vals[0], m_vals[1] + m_vals[2], m_vals[3]]
        l_vals = [n_0, n_1 + n_2, n_3]
        x_labels = ["neither_detected", "one detected", "both detected"]

    fig = go.Figure(
        [
            go.Bar(name = "ISAC Truth", x=x_labels, y=l_vals),
            go.Bar(name = "Model Output", x=x_labels, y=m_vals)
        ]
    )

    fig.update_layout(title="Detection Bins", yaxis_title="Counts", barmode="group")
    fig.show()

# distribution_histograms(preds, labels, combined_mode=True)

# Tests to do! Classwise ECE + MC (later)
# Classwise reliability curves (one class vs rest)
# shap decomp to prob-covariate plot w clopper pearson bounds
# Check Claude for details (Multiclass model evaluation metrics)