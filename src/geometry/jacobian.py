import polars as pl

def add_3d_position_error(
    df: pl.DataFrame,
    range_err_col: str = "sampled_error",
    az_err_col: str = "az_err",
    el_err_col: str = "el_err",
    range_col: str = "range",
    el_col: str = "rx_elevation_deg",
    angle_unit: str = "deg",  # "deg", "rad" or "mrad" for az_err / el_err
) -> pl.DataFrame:
    """Add 3D position error columns computed from range, az and el errors.
 
    Adds these columns, in the same units as `range`:
      err_along   - error along the line of sight (the range error)
      err_cross_h - horizontal cross-range error, R*cos(el)*d_az
      err_cross_v - vertical cross-range error, R*d_el
      err_3d      - exact 3D error: distance between the true point and the
                    perturbed point (accurate for any size of angle error)
      err_3d_approx - small-angle RSS of the three components above
    """
    scale = {"deg": 3.141592653589793 / 180, "rad": 1.0, "mrad": 1e-3}[angle_unit]
 
    R = pl.col(range_col)
    el = pl.col(el_col).radians()          # elevation column is in degrees
    d_r = pl.col(range_err_col)
    d_az = pl.col(az_err_col) * scale
    d_el = pl.col(el_err_col) * scale
 
    # Exact: put the true point at az = 0 (the 3D error doesn't depend on the
    # absolute azimuth), convert both points to Cartesian and take the distance.
    R2, el2 = R + d_r, el + d_el
    dx = R2 * el2.cos() * d_az.cos() - R * el.cos()
    dy = R2 * el2.cos() * d_az.sin()
    dz = R2 * el2.sin() - R * el.sin()
 
    return df.with_columns(
        err_along=d_r,
        err_cross_h=R * el.cos() * d_az,
        err_cross_v=R * d_el,
        err_3d=(dx**2 + dy**2 + dz**2).sqrt(),
    ).with_columns(
        err_3d_approx=(
            pl.col("err_along") ** 2
            + pl.col("err_cross_h") ** 2
            + pl.col("err_cross_v") ** 2
        ).sqrt()
    )