'''
Training (fit) : 

- classify bulk vs non bulk
- simultaneously build bulk + non bulk distributions


inference time:
1. use classifier to get classes
2. use bulk / nonbulk models to sample range error
3. stitch back together
'''
from src.components.base import Component
from src.components.range.regime import RegimeClassifier, GBM_PARAMS
from src.components.range.distributions import BulkModel, NonBulkModel, label_regime
from src.data.load import load_pos, group_split

import numpy as np
import polars as pl


class RangeModel(Component):
    def __init__ (self, class_model = RegimeClassifier(), bulk_model = BulkModel(), non_bulk_model = NonBulkModel()):
        
        self.class_model = class_model
        self.bulk_model = bulk_model
        self.non_bulk_model = non_bulk_model

    def fit(self, train_df, val_df = None, params = GBM_PARAMS):
        out_of_fold, oof_cal, b_final, models = self.class_model.fit(train_df, val_df, params)
        train_df = train_df.with_columns(label_regime(train_df))
        val_df = val_df.with_columns(label_regime(val_df))
        X = train_df.select(["bistatic_range_m", "nn_norm_range_sep"])
        y = train_df.select(["range_error_m", "region_txt"])
        self.bulk_model.fit(X, y)
        self.non_bulk_model.fit(X, y)
        return out_of_fold, oof_cal, b_final, models

    def predict_proba(self, X):
        probs = self.class_model.predict_proba(X)
        return probs

    def sample(self, X, probs, rng):
        X = X.select(["nn_norm_range_sep", "bistatic_range_m"])

        regions = self.class_model.sample(probs, rng)
        regions = regions.with_row_index("_idx")

        df = pl.concat([X, regions], how = "horizontal")

        is_bulk = pl.col("region_txt") == "bulk"
        bulk = df.filter(is_bulk)
        nb = df.filter(~is_bulk)

        bulk_samples = self.bulk_model.sample(bulk, rng)
        nb_samples = self.non_bulk_model.sample(nb, rng)

        bulk_samples = bulk_samples.with_columns(bulk["_idx"])
        nb_samples = nb_samples.with_columns(nb["_idx"])

        samples = (
            pl.concat([bulk_samples, nb_samples], how="vertical")  
            .sort("_idx")
            .drop("_idx")
        )

        return samples

