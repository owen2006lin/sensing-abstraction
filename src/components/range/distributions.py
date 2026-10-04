from src.components.base import Component
from src.data.load import load_pos, group_split
from src.components.range.jitter import jitter
from sklearn.linear_model import LinearRegression
import numpy as np
import polars as pl


REGION_NAMES = ["bulk", "sideband+", "sideband-", "far+", "far-", "near"]   # index = label 0..5


def label_regime(errors):
    e = pl.col("range_error_m")
    labels = (
        errors.select("range_error_m")
        .with_columns(
            pl.when(e.is_between(-0.52, 0.82)).then(0)
            .when(e.is_between(1.40, 1.85)).then(1)
            .when(e.is_between(-2.25, -1.55)).then(2)
            .when(e > 1.85).then(3)
            .when(e < -2.25).then(4)
            .otherwise(5)
            .cast(pl.Int32)
            .alias("region")
        )
        .with_columns(pl.col("region").replace_strict(list(range(6)), REGION_NAMES).alias("region_txt"))
    )
    return labels

tags, labels, features = load_pos()
labels = label_regime(labels)
[X, X_val, y, y_val, t, t_val] = group_split(tags, labels, features, return_tags=True)



def sample_bulk_single(model, bistatic_range, distribution, rng):
    slope = model.coef_
    intercept = model.intercept_

    sampled_residual = rng.choice(distribution)

    p = intercept + slope*bistatic_range + sampled_residual 
    return p


class BulkModel(Component):
    def __init__ (self):
        self.model = LinearRegression()

    def fit(self, X, y):
        # Fit the model
        X = X.select("bistatic_range_m")
        y = y["range_error_m"]

        #some weird mask that interferes with training
        mask = ~((X[:, 0] < 110) & (np.abs(y) < 0.03))
        X = X.filter(mask)
        y = y.filter(mask)

        self.model.fit(X, y)

        # Build residual distribution
        preds = self.model.predict(X)
        residuals = y - preds
        self.distribution = residuals.alias("residual").to_frame()
        return self

    def predict_proba(self):
        print("No probabilities to predict")
        return
    
    def sample(self, X, rng):
        distribution = self.distribution
        outputs = X.with_columns(
            pl.col("bistatic_range_m")
                .map_elements(lambda r: float(np.ravel(sample_bulk_single(self.model, r, distribution, rng))[0]),
                            return_dtype=pl.Float64)
                .alias("sampled_error")
            ).select("sampled_error")
        
        return outputs
'''
b_model = BulkModel()
df = pl.concat([X, y], how = "horizontal")
df = df.filter(pl.col("region_txt") == "bulk")
X1 = df.select("bistatic_range_m")
y1 = df.select("range_error_m")

residuals = b_model.fit(X1, y1)
rng = np.random.default_rng(seed = 42)
samples = b_model.sample(X1, rng)
print(samples)
'''

    
class NonBulkModel(Component):
    def __init__ (self, distributions = None):
        self.distributions = distributions

    def fit(self, X, y, regions = REGION_NAMES[1:]):
        df = pl.concat([X,y], how = "horizontal")
        distributions = []

        for region in regions:
            distr = df.filter(
                pl.col("region_txt") == region
            )
            if region == "far+":
                distr = distr.with_columns(
                    pl.col("nn_norm_range_sep")
                    .cast(pl.Float64)
                    .fill_null(float("inf"))
                    .qcut(5, labels=["1", "2", "3", "4", "5"], allow_duplicates=True)
                    .cast(pl.String)
                    .cast(pl.Int8)
                    .alias("quintile"),
                    pl.col("range_error_m"))
            distributions.append(distr)
        self.distributions = distributions
        return self

    def predict_proba(self):
        print("No probabilities to predict")
        return
    
    def sample(self, df, rng):
        distributions = self.distributions
        df = df.select(["nn_norm_range_sep", "region"]).to_numpy()
        ret = []
        for nn_dist, type in df:
            type = int(type)
            distr = distributions[type-1]
            if type == 3:
                breaks = [distr.get_column("nn_norm_range_sep").quantile(p) for p in (0.2, 0.4, 0.6, 0.8)]
                q = 1 + sum(nn_dist > b for b in breaks)
                quintile_df = distr.filter(pl.col("quintile") == q)
                error = rng.choice(quintile_df["range_error_m"].to_numpy().ravel())
            else:
                error = rng.choice(distr["range_error_m"].to_numpy().ravel())
                error = jitter(error, type)
            ret.append(error)
        ret = pl.DataFrame({"sampled_error" : ret})

        return ret

'''
df = pl.concat([X, y], how = "horizontal")
df = df.filter(pl.col("region_txt") != "bulk")
X = df.select(["nn_norm_range_sep", "range_error_m"])
y = df.select(["region_txt", "region"])
nb_model = NonBulkModel()
distributions = nb_model.fit(X,y)

rng = np.random.default_rng(seed = 42)
df = pl.concat([X,y], how = "horizontal")
samples = nb_model.sample(df, rng)
print(samples)
'''

