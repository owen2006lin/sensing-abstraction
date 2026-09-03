from enum import Enum, auto

class Feature(str, Enum):
    MYFEAT = "myfeat"


class IndepFeature(str, Enum):
  # geometry
  BISTATIC_RANGE_M = "bistatic_range_m"
  LOG_R = "log_R"
  AZ_OFF_BORESIGHT = "az_off_boresight"
  RX_ELEVATION_DEG = "rx_elevation_deg"
  RADIAL_VELOCITY = "radial_velocity"
  ABS_BISTATIC_DOPPLER_HZ = "abs_bistatic_doppler_hz"
  SPEED = "speed"

  # quantization related
  ABS_FRAC_U = "abs_frac_u"
  ABS_FRAC_V = "abs_frac_v"

  # channel / snr
  N_MPC = "n_mpc"
  RSS_MAX = "rss_max"
  RSS_TOTAL_DB = "rss_total_db"
  K_FACTOR_DB = "k_factor_db"
  DOM_EXCESS_DELAY = "dom_excess_delay"
  RMS_DELAY_SPREAD = "rms_delay_spread"

  AOA_SPREAD_W = "aoa_spread_w"
  ZOA_SPREAD_W = "zoa_spread_w"
  FIRST_RSS_MINUS_DOM = "first_rss_minus_dom"

  # multi-target interference
  N_TARGETS = "n_targets"
  NN_NORM_RANGE_SEP = "nn_norm_range_sep"
  NN_NORM_ANG_SEP = "nn_norm_ang_sep"



class PairFeature(str, Enum):
    # geometry
  BISTATIC_RANGE_M = "bistatic_range_m"
  RX_ELEVATION_DEG = "rx_elevation_deg"
  LOG_R = "log_R"
  AZ_OFF_BORESIGHT = "az_off_boresight"
  RADIAL_VELOCITY = "radial_velocity"
  ABS_BISTATIC_DOPPLER_HZ = "abs_bistatic_doppler_hz"
  SPEED = "speed"

  # quantization related
  ABS_FRAC_U = "abs_frac_u"
  ABS_FRAC_V = "abs_frac_v"

  # channel / snr
  N_MPC = "n_mpc"
  RSS_MAX = "rss_max"
  RSS_TOTAL_DB = "rss_total_db"
  K_FACTOR_DB = "k_factor_db"
  DOM_EXCESS_DELAY = "dom_excess_delay"
  RMS_DELAY_SPREAD = "rms_delay_spread"

  AOA_SPREAD_W = "aoa_spread_w"
  ZOA_SPREAD_W = "zoa_spread_w"
  FIRST_RSS_MINUS_DOM = "first_rss_minus_dom"

  # multi-target interference
  N_TARGETS = "n_targets"
  NN_NORM_RANGE_SEP = "nn_norm_range_sep"
  NN_NORM_ANG_SEP = "nn_norm_ang_sep"



class PosFeature(str, Enum):
    # geometry
  BISTATIC_RANGE_M = "bistatic_range_m"
  ABS_BISTATIC_DOPPLER_HZ = "abs_bistatic_doppler_hz"
  LOG_R = "log_R"
  RX_ELEVATION_DEG = "rx_elevation_deg"
  AZ_OFF_BORESIGHT = "az_off_boresight"


  # quantization related
  FRAC_U = "frac_u"
  FRAC_V = "frac_v"
  ABS_FRAC_U = "abs_frac_u"
  ABS_FRAC_V = "abs_frac_v"

  # channel / snr
  N_MPC = "n_mpc"
  RSS_MAX = "rss_max"
  RSS_TOTAL_DB = "rss_total_db"
  K_FACTOR_DB = "k_factor_db"
  DOM_EXCESS_DELAY = "dom_excess_delay"
  RMS_DELAY_SPREAD = "rms_delay_spread"

  AOA_SPREAD_W = "aoa_spread_w"
  ZOA_SPREAD_W = "zoa_spread_w"
  FIRST_RSS_MINUS_DOM = "first_rss_minus_dom"

  # multi-target interference
  N_TARGETS = "n_targets"
  NN_NORM_RANGE_SEP = "nn_norm_range_sep"
  NN_NORM_ANG_SEP = "nn_norm_ang_sep"
