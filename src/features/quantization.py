import polars as pl

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