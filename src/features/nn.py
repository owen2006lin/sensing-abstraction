import polars as pl

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
