import sys
import polars as pl


PATH = "data/raw"
csvs = ["target", "detection", "error", "snr"]
KEYS = ['scenario_id', 'drop_id', 'target_id']


def load_raw(path : str = PATH, names : list[str] = csvs) -> list[pl.DataFrame]:
    dfs = []
    for name in names:
        df = pl.read_csv(f"{path}/{name}.csv")
        dfs.append(df.with_columns(pl.col(pl.Float64, pl.Float32).fill_nan(None)))
    return dfs



# For now, targets don't have an explicit detected/not label which is super annoying
def label_error(error : pl.DataFrame) -> pl.DataFrame:
    error = error.with_columns(
        pl.col("position_error_x_m").is_not_nan().cast(pl.Int8).alias("detected")
    )
    return error

def combine(target: pl.DataFrame, detect : pl.DataFrame, error: pl.DataFrame, snr: pl.DataFrame):
    error = label_error(error)
    snr = build_snr_features(snr)
    combined = target.join(error, on = KEYS, how = "left").join(snr, on = KEYS, how = "left")

    return combined


#-------------------------SNR Features----------------------#
# Organize multiple snr paths into one row of features for each target
def build_snr_features(snr_df : pl.DataFrame) -> pl.DataFrame:
    # Nans might polute computations, keep track of number dropped
    clean = snr_df.filter(pl.col("PathRSS").is_not_null())
    dropped = (snr_df.group_by(KEYS).agg(pl.col("PathRSS").is_null().sum().alias("n_mpc_dropped")))


    # Group by paths corresponding to exact same target
    g = clean.group_by(KEYS, maintain_order = False)

    # Weighted features based off of path power
    power = 10**(pl.col("PathRSS") / 10)

    def _as_expr(x):
        if isinstance(x, str):
            return pl.col(x)
        return x if isinstance(x, pl.Expr) else pl.lit(x)

    def _SC(angle, weight=None):
        """Mean sin/cos of a circular variable given in degrees."""
        a = _as_expr(angle).radians()
        if weight is None:
            return a.sin().mean(), a.cos().mean()
        w = _as_expr(weight)
        return (a.sin() * w).sum() / w.sum(), (a.cos() * w).sum() / w.sum()

    def _R(angle, weight=None):
        """Mean resultant length, optionally power-weighted."""
        S, C = _SC(angle, weight)
        # clamp: R can land at 1+1e-16, making the log positive -> sqrt of a negative
        return (S**2 + C**2).sqrt().clip(1e-12, 1.0)

    def circ_spread(angle, weight=None):
        return (-2 * _R(angle, weight).log()).sqrt().degrees()

    def circ_mean(angle, weight=None):
        S, C = _SC(angle, weight)
        return pl.arctan2(S, C).degrees()

    def safe_std(x):
        return pl.when(pl.len() > 1).then(_as_expr(x).std()).otherwise(0.0)

    f = g.agg(
        # Statistical features about snr path
        pl.len().alias("n_mpc"),
        pl.col("PathRSS").max().alias("rss_max"),
        pl.col("PathRSS").min().alias("rss_min"),
        pl.col("PathRSS").mean().alias("rss_mean"),
        safe_std(pl.col("PathRSS")).alias("rss_std"),
        (10*power.sum().log10()).alias("rss_total_db"),
        pl.col("PathDelay").min().alias("delay_min"),
        pl.col("PathDelay").max().alias("delay_max"),
        (pl.col("PathDelay").max() -  pl.col("PathDelay").min()).alias("delay_spread_raw"),

        circ_spread(pl.col("PathAOA")).alias("aoa_std"),
        safe_std(pl.col("PathZOA")).alias("zoa_std"),


        # Features specific to the max power path
        pl.col("PathDelay").sort_by("PathRSS").last().alias("dom_delay"),
        pl.col("PathZOA").sort_by("PathRSS").last().alias("dom_zoa"),
        circ_mean("PathAOA", power).alias("aoa_mean_circ"),
        (pl.col("PathAOA").get(pl.col("PathRSS").arg_max()) - circ_mean("PathAOA", power)).alias("dom_aoa_offset"),
        pl.col("PathAOA").max().radians().sin().alias("dom_aoa_sin"),
        pl.col("PathAOA").max().radians().cos().alias("dom_aoa_cos"),

    # RSS gap - dom path rss - first path rss
    (pl.col("PathRSS").sort_by("PathRSS").last() - pl.col("PathRSS").sort_by("PathDelay").first()).alias("dom_excess_delay"),

    # Delay Gap, dom path time - first path time
    (pl.col("PathDelay").sort_by("PathRSS").last() - pl.col("PathDelay").sort_by("PathDelay").first()).alias("first_rss_minus_dom")
    )



    def w_spread(col : str):
        # Weighted features based off of path power
        power = 10**(pl.col("PathRSS") / 10)

        w = (pl.col(col) * power).sum() / power.sum()
        return (((pl.col(col) - w)**2 * power).sum() / power.sum()).sqrt()

    w = g.agg(
        # Weighted delay mean
        ((pl.col("PathDelay") * power).sum() / power.sum()).alias("mean_delay_w"),

        w_spread("PathDelay").alias("rms_delay_spread"),
        circ_spread(("PathAOA"), power).alias("aoa_spread_w"),
        w_spread("PathZOA").alias("zoa_spread_w")
    )


    # Rician K factor, dom path vs rest
    p_diffuse = (power.sum() - power.sort_by("PathRSS").last()).clip(lower_bound = 1e-30)
    k = g.agg(
        (pl.col("PathRSS").max() - 10*p_diffuse.log10()).clip(upper_bound=60.0).alias("k_factor_db")
    )

    return f.join(w, on = KEYS, how = "left").join(k, on = KEYS, how = "left").join(dropped, on = KEYS, how = "left")


#------------------------Quantization Features--------------#
AZ0 = 30.0
NGRID = 32.0

def uv_exprs(az_col : str, el_col: str) -> tuple[pl.Expr, pl.Expr]:
    el_rad = pl.col(el_col).radians()
    az_rel_rad = (pl.col(az_col) - AZ0).radians()

    u = (el_rad.cos() * az_rel_rad.sin())
    v = (el_rad.sin())
    return u,v


def inv_uv(u_col : str, v_col : str) -> tuple[pl.Expr, pl.Expr]:
    v_clip = pl.col(v_col).clip(-1.0,1.0)
    el_rad = v_clip.arcsin()

    ce = el_rad.cos()
    ce_safe = pl.when(ce < 1e-9).then(1e-9).otherwise(ce)
    az_deg = AZ0 + (pl.col(u_col)/ce_safe).clip(-1.0,1.0).arcsin().degrees()

    return az_deg, el_rad.degrees()

# features relating to either u or v
# Note: this depends on both target and detected so the two should be merged
def build_angle_features(df : pl.DataFrame) -> pl.DataFrame:
    u_true_expr, v_true_expr = uv_exprs("rx_azimuth_deg","rx_elevation_deg")
    u_expr, v_expr = uv_exprs("estimated_rx_azimuth_deg", "estimated_rx_elevation_deg")

    # UV values, estimated and true
    uv_feats = df.select(
        u_expr.alias("u_coord"),
        v_expr.alias("v_coord"),
        u_true_expr.alias("u_true"),
        v_true_expr.alias("v_true"),


        #integer grid value
        (NGRID*u_true_expr).round().alias("ku_true"),
        (NGRID*v_true_expr).round().alias("kv_true"),
        (NGRID*u_expr).round().alias("ku_est"),
        (NGRID*v_expr).round().alias("kv_est"),

    ).with_columns(
        # Fractional offset from nearest grid cell
        (NGRID * pl.col("u_true") - pl.col("ku_true")).alias("frac_u"),
        (NGRID * pl.col("v_true") - pl.col("kv_true")).alias("frac_v"),

        # "slip" - discrete offset... how many cells away you've slipped
        (pl.col("ku_est") - pl.col("ku_true")).alias("slip_u"),
        (pl.col("kv_est") - pl.col("kv_true")).alias("slip_v")

    ).with_columns(
        # Absolute fractional offset
        pl.col("frac_u").abs().alias("abs_frac_u"),
        pl.col("frac_v").abs().alias("abs_frac_v")
    )

    return uv_feats

#------------------------Geometry Related Features--------------#
tangential_velocity_expr = (
    (
        pl.col("target_v_x_ms").pow(2)
        + pl.col("target_v_y_ms").pow(2)
        + pl.col("target_v_z_ms").pow(2)
        - pl.col("radial_velocity").pow(2)
    )
    .clip(lower_bound=0)
    .sqrt()
    .alias("tangential_velocity")
)
def build_geometry_features(df : pl.DataFrame) -> pl.DataFrame:

    geo = df.select(
        (pl.col("estimated_bistatic_range_m") - pl.col("bistatic_range_m")).alias("range_err_bi"),
        pl.col("rx_target_distance_m").log().alias("log_R"),
        (pl.col("target_v_x_ms")**2 + pl.col("target_v_x_ms")**2).alias("speed"),
        (pl.col("rx_azimuth_deg") - AZ0).alias("az_off_boresight"),
        pl.col("bistatic_doppler_hz").abs().alias("abs_bistatic_doppler_hz"),
        # radar equation features
        (pl.col("tx_target_distance_m")*pl.col("rx_target_distance_m")).log10().alias("log_range_product"),
        (-20*(pl.col("tx_target_distance_m").log10()) - (20*(pl.col("rx_target_distance_m").log10()))).alias("path_loss_proxy"),
        tangential_velocity_expr
    )
    return geo

#------------------------Nearest Neighbor Features--------------#
def build_nn_features(df : pl.DataFrame) -> pl.DataFrame:
    nn_features = df.select(
        pl.col("nearest_neighbor_xyz_distance_m").fill_nan(1e4).alias("nn_dist"),
        pl.col("nearest_neighbor_bistatic_range_sep_m").abs().fill_nan(1e4).alias("nn_range_sep"),
        pl.col("nearest_neighbor_rx_az_sep_deg").abs().fill_nan(180.).alias("nn_az_sep"),
        pl.col("nearest_neighbor_rx_el_sep_deg").abs().fill_nan(180.).alias("nn_el_sep"),
        pl.col("nearest_neighbor_norm_range_sep").fill_nan(1e3).alias("nn_norm_range_sep"),
        pl.col("nearest_neighbor_norm_rx_angle_sep").fill_nan(1e3).alias("nn_norm_ang_sep"),
        pl.col("num_targets_in_sample").alias("n_targets")
    )
    return nn_features



#-----------------------Example Usage---------------------#
[target, detect, error, snr] = load_raw()
error = label_error(error)
snr_features = build_snr_features(snr)
nn_features = build_nn_features(target)


# some changes to detection before merging, changing column name "associated_target_id" -> "target_id"
# and also dropping false positives for now
detect = detect.rename({"associated_target_id" : "target_id"})
detect = detect.drop_nans("target_id").with_columns(
    pl.col("target_id").cast(pl.Int64)
)
combined = target.join(detect, on = KEYS, how = "left")
angle_features = build_angle_features(combined)
geo_features = build_geometry_features(combined)
# Completed dataframe with all features and labels, to cache
merged_raw = (target
                .join(detect, on = KEYS, how = "left")
                .join(error, on = KEYS, how = "left")
                .join(snr_features, on = KEYS, how = "left"))

merged = pl.concat([merged_raw, geo_features, angle_features, nn_features], how = "horizontal")
merged.write_csv("data/feature_cache/all_labels_features.csv")
