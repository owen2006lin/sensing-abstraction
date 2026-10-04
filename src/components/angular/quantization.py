from src.components.angular.slip import SlipModel
from src.feature_building.quantization import inv_uv
from src.data.load import load_pos, group_split
import polars as pl
import numpy as np

NGRID = 32.0


class AngleModel():
    def __init__(self, model = None):
        if not model:
            self.slip_model = SlipModel()
        else:
            self.slip_model = model

    def fit(self, X, y, print_params = False):
        self.slip_model.fit(X, y, print_params)
        return self
    
    def predict_proba(self, X):
        prob_u, prob_v = self.slip_model.predict_proba(X)
        return prob_u, prob_v
    
    def sample(self, X, p_u, p_v, rng):
        slips = self.slip_model.sample(X, p_u, p_v, rng)

        df = pl.concat([X, slips], how = "horizontal")
        df = df.select(
            pl.col("u_true"),
            pl.col("v_true"),
            pl.col("rx_azimuth_deg"),
            pl.col("rx_elevation_deg"),
            ((NGRID * pl.col("u_true") - pl.col("frac_u") + pl.col("slip_u")) / NGRID).alias("u_est"),
            ((NGRID * pl.col("v_true") - pl.col("frac_v") + pl.col("slip_v")) / NGRID).alias("v_est"),
        )

        az_est, el_est = inv_uv("u_est", "v_est")
        df = df.with_columns(
            az_est.alias("az_est"),
            el_est.alias("el_est"),
            (pl.col("u_true") - pl.col("u_est")).alias("u_error"),   # = (frac - slip)/NGRID
            (pl.col("v_true") - pl.col("v_est")).alias("v_error"),
        ).with_columns(
            # wrap az error into [-180, 180)
            (((pl.col("az_est") - pl.col("rx_azimuth_deg") + 180) % 360) - 180).alias("az_err"),
            (pl.col("el_est") - pl.col("rx_elevation_deg")).alias("el_err"),
        )

        return df

tags, labels, features = load_pos()
[X, X_val, y, y_val, t, t_val] = group_split(tags, labels, features, return_tags=True)
model = AngleModel()
model.fit(X,y, print_params=True)
u,v = model.predict_proba(X_val)
samples = model.sample(X_val, u, v, rng = np.random.default_rng(seed = 42))
