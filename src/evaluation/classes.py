from src.components.base import Component
from src.data.features import IndepFeature
import numpy as np
import polars as pl

# Measurement space errors the baselines sample for detected points (columns from load_pos())
ERROR_COLS = ["range_error_m", "rx_azimuth_error_deg", "rx_elevation_error_deg"]

#====================================B1 : Constant Rate + Gaussian Error=======================#
# One detection rate for every point, errors drawn from a single multivariate gaussian fit on detected points
# y : DataFrame with "detected" and optionally error columns (null where missed)

class ConstantClassifier(Component):
    def __init__(self, rate = None, mean = None, cov = None, error_cols = None):
        self.rate = rate
        self.mean = mean
        self.cov = cov
        self.error_cols = error_cols if error_cols is not None else []

    def fit(self, X, y, X_val = None, y_val = None, params = None, callbacks = None):
        self.rate = y["detected"].mean()

        self.error_cols = [c for c in y.columns if c != "detected"]
        if self.error_cols:
            errors = y.filter(pl.col("detected") == 1).select(self.error_cols).drop_nulls().to_numpy()
            self.mean = errors.mean(axis = 0)
            self.cov = np.cov(errors, rowvar = False)
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.rate)

    def sample(self, X, rng, features = None):
        X = np.asarray(X)
        detected = rng.uniform(size = X.shape) < X
        df = pl.DataFrame({"detected" : detected})

        if self.error_cols:
            errors = rng.multivariate_normal(self.mean, self.cov, size = len(X))
            errors[~detected] = np.nan
            df = df.with_columns(pl.DataFrame(errors, schema = self.error_cols).fill_nan(None))
        return df



#====================================B4 : Mixture Density Network=======================#
# MLP trunk with two heads : P(detected) (sigmoid) and a K component diagonal gaussian mixture over the errors
# of detected points. Written in numpy (manual backprop + Adam) so no torch dependency is needed
# Loss = BCE(detected) + error_weight * mixture NLL (detected rows with errors only)

MDN_PARAMS = {
    "hidden" : 64,
    "n_components" : 5,
    "learning_rate" : 1e-3,
    "batch_size" : 512,
    "epochs" : 200,
    "patience" : 20,       # stops after 20 epochs of no val improvement
    "error_weight" : 1.0,
    "log_period" : 10,
    "seed" : 42,
}

LOG_2PI = np.log(2 * np.pi)

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _log_softmax(x):
    x = x - x.max(axis = 1, keepdims = True)
    return x - np.log(np.exp(x).sum(axis = 1, keepdims = True))


class MDNClassifier(Component):
    def __init__(self, params = MDN_PARAMS):
        self.params = params
        self.weights = None
        self.feature_names = [f.value for f in IndepFeature]
        self.error_cols = []

    # Standardize, and add a missing indicator for any column with nulls (eg nn_norm_range_sep with no neighbor)
    def _prep(self, X, fit = False):
        X = X.select(self.feature_names)
        if fit:
            self.null_cols = [c for c in X.columns if X[c].null_count() > 0]
            self.x_mean = X.mean().to_numpy().ravel()
            self.x_std = X.std().to_numpy().ravel() + 1e-8

        missing = X.select(self.null_cols).select(pl.all().is_null().cast(pl.Float64)).to_numpy()
        Z = (X.to_numpy().astype(float) - self.x_mean) / self.x_std
        Z = np.nan_to_num(Z, nan = 0.0)
        return np.hstack([Z, missing])

    def _prep_y(self, y, fit = False):
        det = y["detected"].to_numpy().astype(float)
        D = len(self.error_cols)
        if D == 0:
            return det, np.zeros((len(det), 0)), np.zeros(len(det), dtype = bool)

        err = y.select(self.error_cols).to_numpy().astype(float)
        mask = (det == 1) & ~np.isnan(err).any(axis = 1)
        if fit:
            self.e_mean = err[mask].mean(axis = 0)
            self.e_std = err[mask].std(axis = 0) + 1e-8
        err = np.nan_to_num((err - self.e_mean) / self.e_std, nan = 0.0)
        return det, err, mask

    def _init_weights(self, d, rng):
        h, K, D = self.params["hidden"], self.params["n_components"], max(len(self.error_cols), 1)
        self.weights = {
            "W1" : rng.normal(0, 1 / np.sqrt(d), (d, h)), "b1" : np.zeros(h),
            "W2" : rng.normal(0, 1 / np.sqrt(h), (h, h)), "b2" : np.zeros(h),
            "Wd" : rng.normal(0, 1 / np.sqrt(h), (h, 1)), "bd" : np.zeros(1),
            "Wm" : rng.normal(0, 0.1 / np.sqrt(h), (h, K * (1 + 2 * D))), "bm" : np.zeros(K * (1 + 2 * D)),
        }

    def _forward(self, Z):
        w = self.weights
        K, D = self.params["n_components"], max(len(self.error_cols), 1)
        a1 = np.tanh(Z @ w["W1"] + w["b1"])
        a2 = np.tanh(a1 @ w["W2"] + w["b2"])
        logit = (a2 @ w["Wd"] + w["bd"]).ravel()

        m = a2 @ w["Wm"] + w["bm"]
        pi_logit = m[:, :K]
        mu = m[:, K : K + K * D].reshape(-1, K, D)
        log_sig = np.clip(m[:, K + K * D:].reshape(-1, K, D), -7, 7)
        return a1, a2, logit, pi_logit, mu, log_sig

    def _loss_grads(self, Z, det, err, mask):
        w = self.weights
        n = len(Z)
        w_err = self.params["error_weight"]
        a1, a2, logit, pi_logit, mu, log_sig = self._forward(Z)

        # BCE
        p = np.clip(_sigmoid(logit), 1e-7, 1 - 1e-7)
        bce = -np.mean(det * np.log(p) + (1 - det) * np.log(1 - p))
        dlogit = (p - det) / n

        # mixture NLL, only on detected rows with errors
        dpi = np.zeros_like(pi_logit)
        dmu = np.zeros_like(mu)
        dls = np.zeros_like(log_sig)
        nll = 0.0
        m = mask.sum()
        if len(self.error_cols) and m > 0:
            diff = err[:, None, :] - mu
            inv_var = np.exp(-2 * log_sig)
            log_n = np.sum(-0.5 * diff**2 * inv_var - log_sig - 0.5 * LOG_2PI, axis = 2)
            log_pi = _log_softmax(pi_logit)
            lj = log_pi + log_n
            lse = lj.max(axis = 1) + np.log(np.exp(lj - lj.max(axis = 1, keepdims = True)).sum(axis = 1))
            gamma = np.exp(lj - lse[:, None])

            nll = -np.sum(lse * mask) / m
            scale = (w_err * mask / m)[:, None]
            dpi = (np.exp(log_pi) - gamma) * scale
            dmu = -gamma[..., None] * diff * inv_var * scale[..., None]
            dls = -gamma[..., None] * (diff**2 * inv_var - 1) * scale[..., None]

        dm = np.hstack([dpi, dmu.reshape(n, -1), dls.reshape(n, -1)])

        # backprop through trunk
        da2 = dlogit[:, None] @ w["Wd"].T + dm @ w["Wm"].T
        dz2 = da2 * (1 - a2**2)
        dz1 = (dz2 @ w["W2"].T) * (1 - a1**2)
        grads = {
            "Wd" : a2.T @ dlogit[:, None], "bd" : dlogit.sum(keepdims = True),
            "Wm" : a2.T @ dm, "bm" : dm.sum(axis = 0),
            "W2" : a1.T @ dz2, "b2" : dz2.sum(axis = 0),
            "W1" : Z.T @ dz1, "b1" : dz1.sum(axis = 0),
        }
        return bce + w_err * nll, bce, nll, grads

    # X : features DataFrame (needs IndepFeature columns), y : "detected" + optional error columns
    def fit(self, X, y, X_val = None, y_val = None, params = None, callbacks = None):
        if params is not None:
            self.params = params
        rng = np.random.default_rng(seed = self.params["seed"])

        self.error_cols = [c for c in y.columns if c != "detected"]
        Z = self._prep(X, fit = True)
        det, err, mask = self._prep_y(y, fit = True)
        has_val = X_val is not None and y_val is not None
        if has_val:
            Z_val = self._prep(X_val)
            det_val, err_val, mask_val = self._prep_y(y_val)

        self._init_weights(Z.shape[1], rng)
        m_t = {k : np.zeros_like(v) for k, v in self.weights.items()}
        v_t = {k : np.zeros_like(v) for k, v in self.weights.items()}
        b1, b2, lr, t = 0.9, 0.999, self.params["learning_rate"], 0

        best, best_weights, stale = np.inf, None, 0
        for epoch in range(self.params["epochs"]):
            order = rng.permutation(len(Z))
            for start in range(0, len(Z), self.params["batch_size"]):
                idx = order[start : start + self.params["batch_size"]]
                _, _, _, grads = self._loss_grads(Z[idx], det[idx], err[idx], mask[idx])

                # Adam
                t += 1
                for k in self.weights:
                    m_t[k] = b1 * m_t[k] + (1 - b1) * grads[k]
                    v_t[k] = b2 * v_t[k] + (1 - b2) * grads[k]**2
                    m_hat = m_t[k] / (1 - b1**t)
                    v_hat = v_t[k] / (1 - b2**t)
                    self.weights[k] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            loss, bce, nll, _ = self._loss_grads(Z, det, err, mask)
            msg = f"[{epoch}] train loss {loss:.4f} (bce {bce:.4f}, nll {nll:.4f})"
            if has_val:
                val_loss, val_bce, val_nll, _ = self._loss_grads(Z_val, det_val, err_val, mask_val)
                msg += f"  valid loss {val_loss:.4f} (bce {val_bce:.4f}, nll {val_nll:.4f})"

                if val_loss < best:
                    best, stale = val_loss, 0
                    best_weights = {k : v.copy() for k, v in self.weights.items()}
                else:
                    stale += 1
                    if stale >= self.params["patience"]:
                        print(f"Early stopping, best valid loss {best:.4f}")
                        break

            if epoch % self.params["log_period"] == 0:
                print(msg)

        if best_weights is not None:
            self.weights = best_weights
        return self

    def predict_proba(self, X):
        _, _, logit, _, _, _ = self._forward(self._prep(X))
        return _sigmoid(logit)

    # X : probs from predict_proba. Pass features as well to also sample errors from the mixture
    def sample(self, X, rng, features = None):
        X = np.asarray(X)
        detected = rng.uniform(size = X.shape) < X
        df = pl.DataFrame({"detected" : detected})

        if self.error_cols and features is not None:
            _, _, _, pi_logit, mu, log_sig = self._forward(self._prep(features))
            pi = np.exp(_log_softmax(pi_logit))
            k = (pi.cumsum(axis = 1) > rng.uniform(size = (len(pi), 1))).argmax(axis = 1)

            rows = np.arange(len(pi))
            errors = mu[rows, k] + np.exp(log_sig[rows, k]) * rng.standard_normal(mu[rows, k].shape)
            errors = errors * self.e_std + self.e_mean
            errors[~detected] = np.nan
            df = df.with_columns(pl.DataFrame(errors, schema = self.error_cols).fill_nan(None))
        return df
