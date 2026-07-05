#!/usr/bin/env python3
"""M6: slip-detector study. Induced-slip trials (setpoint ramped down until
the box drops) provide positives with kinematic ground-truth slip times;
clean holds provide negatives. The vibration detector is calibrated on a
training split and scored on the test split, at the ideal sensor rate and at
the TacTip camera rate.

  python scripts/eval_slip.py            # ~10–20 min, sequential
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.params import BoxSpec, TrialSpec, deg
from pinchlab.slipdetect import (DetectorConfig, band_energy, detect,
                                 downsample, score_trials, tangential_signal)
from pinchlab.trial import get_kinematics, run_trial
from pinchlab import plots

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")


def truth_time(res) -> float:
    """Kinematic slip truth: incipient sliding if detected, else the drift
    threshold crossing."""
    if not np.isnan(res.slip_time):
        return res.slip_time
    s = res.series
    over = np.nonzero(np.asarray(s["drift"]) > res.spec.drift_threshold)[0]
    if len(over):
        return float(np.asarray(s["t"])[over[0]])
    return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--f0", type=float, default=0.8)
    ap.add_argument("--ramp", type=float, default=0.15, help="N/s")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tactip-hz", type=float, default=100.0)
    args = ap.parse_args()

    kin = get_kinematics()
    postures = []
    for w, p in [(0.0, 0.0), (7.5, 4.0)]:
        posture, _ = kin.solve_mcp_for_gap(deg(w), deg(p), 0.024)
        postures.append((w, p, posture))

    cache_dir = os.path.join(RESULTS, "slip_episodes")
    os.makedirs(cache_dir, exist_ok=True)
    keep_keys = ("t", "f2_L", "f2_R", "f3_L", "f3_R", "drift")

    episodes = []   # (series, truth, kind, w, p, seed, outcome)
    for w, p, posture in postures:
        for seed in range(args.seeds):
            for kind, ramp in (("pos", args.ramp), ("neg", 0.0)):
                key = f"ep_{kind}_w{w:+05.1f}_p{p:+05.1f}_s{seed}"
                path = os.path.join(cache_dir, key + ".npz")
                if os.path.exists(path):
                    d = np.load(path, allow_pickle=False)
                    series = {k: d[k] for k in keep_keys}
                    truth = float(d["truth"])
                    outcome = str(d["outcome"])
                else:
                    spec = TrialSpec(posture=posture, box=BoxSpec(),
                                     force_setpoint=args.f0, setpoint_ramp=ramp,
                                     time_step=1e-3, log_hz=1000.0, seed=seed,
                                     tag=f"slip_{kind}")
                    res = run_trial(spec, keep_series=True)
                    truth = truth_time(res) if kind == "pos" else (
                        np.nan if res.held else truth_time(res))
                    outcome = res.outcome
                    series = {k: np.asarray(res.series[k]) for k in keep_keys}
                    np.savez_compressed(path, truth=truth, outcome=outcome,
                                        **series)
                episodes.append((series, truth, kind, w, p, seed, outcome))
                print(f"w={w:+5.1f} p={p:+4.1f} seed={seed} {kind}: "
                      f"outcome={outcome} truth={truth}", flush=True)

    # Stratified split: both halves get positives AND negatives.
    pos = [(s, tr) for s, tr, kind, *_ in episodes if kind == "pos"]
    neg = [(s, tr) for s, tr, kind, *_ in episodes if kind == "neg"]
    train = pos[0::2] + neg[0::2]
    test = pos[1::2] + neg[1::2]

    cfg = DetectorConfig(tactip_hz=args.tactip_hz)
    report = {}
    for rate in ("ideal", "tactip"):
        cal = score_trials(train, cfg, rate=rate, threshold=None)
        scored = score_trials(test, cfg, rate=rate, threshold=cal["threshold"])
        scored["threshold_from_train"] = cal["threshold"]
        report[rate] = {k: v for k, v in scored.items() if k != "det_times"}
        print(f"\n[{rate}] thr={cal['threshold']:.3g} N²  "
              f"tp={scored['tp']} fp={scored['fp']} fn={scored['fn']} "
              f"tn={scored['tn']}  latency mean={scored['latency_mean']:.3f}s "
              f"max={scored['latency_max']:.3f}s")

    # Example figure: first positive test episode, both rates.
    os.makedirs(RESULTS, exist_ok=True)
    pos = next(e for e in episodes if e[2] == "pos")
    s, truth = pos[0], pos[1]
    for rate in ("ideal", "tactip"):
        t, x = tangential_signal(s, "L")
        _, xr = tangential_signal(s, "R")
        x = np.maximum(x, xr)
        if rate == "tactip":
            t, x = downsample(t, x, cfg.tactip_hz)
        thr = report[rate]["threshold_from_train"]
        e = band_energy(t, x, cfg)
        td = detect(t, x, cfg, thr)
        plots.plot_slip_detection(
            t, x, e, thr, truth, td,
            f"slip detection — {rate} rate "
            f"({'full sim rate' if rate == 'ideal' else f'{cfg.tactip_hz:.0f} Hz TacTip'})",
            os.path.join(RESULTS, f"slip_detection_{rate}.png"))

    with open(os.path.join(RESULTS, "slip_eval.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nreport → results/slip_eval.json, figures → results/slip_detection_*.png")


if __name__ == "__main__":
    main()
