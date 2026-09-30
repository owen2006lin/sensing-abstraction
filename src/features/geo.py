import polars as pl

AZ0 = 30.0
NGRID = 32.0


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
