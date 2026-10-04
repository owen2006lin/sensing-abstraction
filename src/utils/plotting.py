import polars as pl
import plotly.graph_objects as go

def plot_hist(vals, names):
    fig = go.Figure()
    
    for i in range(len(vals)):
        lo = min(map(min, vals)); hi = max(map(max, vals))
        fig.add_trace(go.Histogram(
            x = vals[i],
            xbins=dict(start=lo, end=hi, size=(hi - lo) / 100),
            name = names[i],
            opacity = 0.6,
            bingroup=1
        ))
    
    fig.update_layout(
        barmode="overlay",
        xaxis_title="value",
        yaxis_title="count",
        legend_title="series"
    )
    
    fig.show()