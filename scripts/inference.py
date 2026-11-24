import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import argparse
import numpy as np
import polars as pl

# Times the trained sensing abstraction end to end on the held out test set (same scene grouped split as scripts/train.py).
# The input is what a system level simulator would hand the abstraction : target geometry (target.csv) and the channel
# multipath components (snr.csv). Everything downstream of that is timed :
#   features  : SNR / multipath aggregation, geometry, angle grid and nearest neighbour features
#   detection : FullClassifier probabilities, Bernoulli / categorical draws, reassembly onto targets
#   error     : FullPositionModel range + angle draws and the 3D position error, for detected targets only
# Run from the repo root after scripts/train.py : python scripts/inference.py --runs 10
from src.data.load import load_raw, group_split, KEYS
from src.feature_building.snr import build_snr_features
from src.feature_building.nn import build_nn_features
from src.feature_building.geo import build_geometry_features
from src.feature_building.quantization import build_angle_features
from src.data.features import IndepFeature, PairFeature, PosFeature
from src.components.detection import FullClassifier
from src.components.position import FullPositionModel

CLASS_PATH = "models/full_classifier.joblib"
POS_PATH = "models/position_model.joblib"
OUT = Path("results/inference")
# build_geometry_features / build_angle_features also derive label columns from the detector's estimates,
# which do not exist at inference time, so they are filled with nulls (only the true geometry columns are used as features)
EST_COLS = ["estimated_bistatic_range_m", "estimated_rx_azimuth_deg", "estimated_rx_elevation_deg"]
FEATURE_COLS = list(dict.fromkeys([f.value for f in [*IndepFeature, *PairFeature, *PosFeature]]))


def test_split(target, snr):
    # Reproduce the outer split of scripts/train.py : group by (scenario_id, drop_id), 20% test, seed 42
    tags = target.select(KEYS)
    _, test_tags, _, _ = group_split(tags, tags, tags)
    scenes = test_tags.select(["scenario_id", "drop_id"]).unique()
    return (target.join(scenes, on = ["scenario_id", "drop_id"], how = "semi"),
            snr.join(scenes, on = ["scenario_id", "drop_id"], how = "semi"))


def build_features(target, snr):
    snr_feats = build_snr_features(snr)
    target = target.with_columns([pl.lit(None, dtype = pl.Float64).alias(c) for c in EST_COLS])
    geo = build_geometry_features(target)
    angle = build_angle_features(target)
    nn = build_nn_features(target)
    df = pl.concat([target.select(KEYS), geo, angle, nn], how = "horizontal_extend")
    df = df.join(snr_feats, on = KEYS, how = "left")
    return df.with_columns(target["bistatic_range_m"], target["rx_elevation_deg"], target["rx_azimuth_deg"],
                           target["radial_velocity"]).select([*KEYS, *FEATURE_COLS])


def run_once(target, snr, classifier, pos_model, rng):
    t = {}
    t0 = time.perf_counter()
    df = build_features(target, snr)
    t["features"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    indep_preds, pair_preds = classifier.predict_proba(df)
    indep_s, pair_s = classifier.sample(indep_preds, pair_preds, rng)
    df = classifier.reassemble(df, indep_s, pair_s)
    t["detection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    detected = df.filter(pl.col("detected"))
    errors = pos_model.sample(detected.select([f.value for f in PosFeature]), rng)
    t["error"] = time.perf_counter() - t0

    t["total"] = t["features"] + t["detection"] + t["error"]
    t["n_targets"] = df.height
    t["n_detected"] = detected.height
    return t, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type = int, default = 10)
    parser.add_argument("--warmup", type = int, default = 1)
    args = parser.parse_args()

    target, _, _, snr = load_raw(names = ["target", "detection", "error", "snr"])
    target, snr = test_split(target, snr)
    n_scenes = target.select(["scenario_id", "drop_id"]).n_unique()

    t0 = time.perf_counter()
    classifier = FullClassifier.load(CLASS_PATH)
    pos_model = FullPositionModel.load(POS_PATH)
    load_time = time.perf_counter() - t0
    print(f"test set : {target.height} targets, {snr.height} multipath components, {n_scenes} drops")
    print(f"model load : {load_time:.3f} s")

    rng = np.random.default_rng(seed = 42)
    for _ in range(args.warmup):
        run_once(target, snr, classifier, pos_model, rng)

    rows = []
    for i in range(args.runs):
        t, errors = run_once(target, snr, classifier, pos_model, rng)
        rows.append({"run" : i, **t})
        print(f"run {i} : features {t['features']:.3f} s, detection {t['detection']:.3f} s, "
              f"error {t['error']:.3f} s, total {t['total']:.3f} s ({t['n_detected']} / {t['n_targets']} detected)")

    timing = pl.DataFrame(rows)
    OUT.mkdir(parents = True, exist_ok = True)
    timing.write_csv(OUT / "timing.csv")

    stages = ["features", "detection", "error", "total"]
    total = timing["total"].mean()
    lines = [f"test set : {target.height} targets, {snr.height} multipath components, {n_scenes} drops, {args.runs} runs",
             f"model load (one off) : {load_time:.3f} s"]
    lines += [f"{s:<10} : mean {timing[s].mean():.4f} s, std {(timing[s].std() or 0.0):.4f} s, "
              f"min {timing[s].min():.4f} s, max {timing[s].max():.4f} s" for s in stages]
    lines += [f"per target : {1e6 * total / target.height:.1f} us, per drop : {1e3 * total / n_scenes:.3f} ms"]
    summary = "\n".join(lines)
    (OUT / "summary.md").write_text(f"# Inference timing\n\n```\n{summary}\n```\n")
    print("\n" + summary)
