from src.data.load import load_pos, group_split
from src.components.range.range import RangeModel
from src.components.angular.quantization import AngleModel
from src.components.base import Component
from src.components.range.regime import GBM_PARAMS
from src.geometry.jacobian import add_3d_position_error
import polars as pl
import numpy as np

tags, labels, features = load_pos()
[X, X_val, y, y_val, t, t_val] = group_split(tags, labels, features, return_tags=True)
train_df = pl.concat([X, y, t], how = "horizontal")
val_df = pl.concat([X_val, y_val, t_val], how = "horizontal")

class FullPositionModel(Component):
    def __init__ (self, range_model = RangeModel(), angle_model = AngleModel()):
        self.range_model = range_model
        self.angle_model = angle_model

    def fit(self, train_df, val_df = None, params = GBM_PARAMS):
        y = train_df.select(["slip_v", "slip_u"])
        self.angle_model.fit(train_df, y, print_params = True)
        if val_df is None:
            self.range_model.fit(train_df, params)
        else:
            self.range_model.fit(train_df, val_df, params)
        return self

    def predict_proba(self):
        print("Delegate probability prediction to individual angle or range models")
        return

    def sample(self, df, rng):
        range_probs = self.range_model.predict_proba(df)
        range_samples = self.range_model.sample(df, range_probs, rng)

        prob_u, prob_v = self.angle_model.predict_proba(df)
        angle_samples = self.angle_model.sample(df, prob_u, prob_v, rng)

        df_feats = (df.select(["rx_elevation_deg", "bistatic_range_m"])
                    .with_columns((pl.col("bistatic_range_m") / 2).alias("range")))
        angle_errors = angle_samples.select(["el_err", "az_err"])
        ret = pl.concat([df_feats, range_samples, angle_errors], how = "horizontal")
        pos_error = add_3d_position_error(ret)
        return pos_error





