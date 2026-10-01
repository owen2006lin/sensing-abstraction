import polars as pl
KEYS = ['scenario_id', 'drop_id', 'target_id']


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

