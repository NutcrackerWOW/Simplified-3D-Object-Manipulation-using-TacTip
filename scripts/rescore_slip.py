#!/usr/bin/env python3
"""Offline re-scoring of the expanded slip study (no re-simulation).

The global 3x-max-floor calibration fails on the 15-condition envelope
(tp=0/fn=30): the noisiest training episode sets a threshold above the slip
burst of quieter conditions. This script quantifies why and tests fixes:

  a) per-episode separability: slip-burst peak energy vs the episode's own
     pre-slip floor — is slip detectable at all, per condition?
  b) per-condition calibration: threshold from the SAME condition's training
     episodes only (1 pos + 1 neg each), scored on that condition's test eps
     (diagnostic upper bound — assumes training data at every condition).
  c) best-achievable global threshold (oracle sweep on the test set — upper
     bound for any global calibration).
  d) SCHEDULED threshold (deployable, primary): log-floor fitted over
     (wave, pip, mass) from the training floors — same recipe as the
     grip-force map — threshold = 3 x predicted floor at any posture/load.
  e) ADAPTIVE threshold (deployable, supplement): k x a causal running
     estimate of the local floor (trailing block percentile with a guard
     gap); k chosen on the training split. Needs no model and no posture
     input, at the cost of a warm-up period.

  python scripts/rescore_slip.py
"""

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.fit import basis
from pinchlab.slipdetect import DetectorConfig, band_energy, downsample

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")
CACHE = os.path.join(RESULTS, "slip_episodes_big")
PAT = re.compile(r"ep_(pos|neg)_w(.+)_p(.+)_m(\d+)_s(\d+)\.npz")


def load_episodes():
    eps = []
    for path in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        m = PAT.match(os.path.basename(path))
        if not m:
            continue
        kind, w, p, mass, seed = (m.group(1), float(m.group(2)),
                                  float(m.group(3)), int(m.group(4)),
                                  int(m.group(5)))
        d = np.load(path, allow_pickle=False)
        t = d["t"]
        x = np.maximum(np.hypot(d["f2_L"], d["f3_L"]),
                       np.hypot(d["f2_R"], d["f3_R"]))
        eps.append(dict(kind=kind, wave=w, pip=p, mass=mass, seed=seed,
                        cond=(w, p, mass), t=t, x=x,
                        truth=float(d["truth"])))
    return eps


def energies(eps, cfg, rate):
    """Attach band-energy series per episode for one sensor rate."""
    out = []
    for e in eps:
        t, x = e["t"], e["x"]
        if rate == "tactip":
            t, x = downsample(t, x, cfg.tactip_hz)
        out.append(dict(e, t=t, e=band_energy(t, x, cfg)))
    return out


def floor_of(ep):
    t_end = ep["truth"] if not np.isnan(ep["truth"]) else ep["t"][-1]
    pre = ep["e"][ep["t"] < t_end - 0.05]
    return float(np.percentile(pre, 99)) if len(pre) else np.nan


def classify(ep, thr, pre_tol=0.3):
    above = np.nonzero(ep["e"] > thr)[0]
    td = float(ep["t"][above[0]]) if len(above) else np.nan
    truth = ep["truth"]
    if np.isnan(truth):
        return ("tn" if np.isnan(td) else "fp"), np.nan
    if np.isnan(td):
        return "fn", np.nan
    if td >= truth - pre_tol:
        return "tp", td - truth
    return "fp", np.nan


def score(eps, thr_fn):
    """thr_fn(ep) -> threshold for that episode."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    lats = []
    for ep in eps:
        c, lat = classify(ep, thr_fn(ep))
        counts[c] += 1
        if c == "tp":
            lats.append(lat)
    counts["latency_mean"] = round(float(np.mean(lats)), 4) if lats else None
    counts["latency_max"] = round(float(np.max(lats)), 4) if lats else None
    return counts


def fit_floor_model(train):
    """Least-squares fit of log10(pre-slip floor) over posture + mass —
    the deployable scheduled-threshold model."""
    X = np.column_stack([
        basis(np.deg2rad([e["wave"] for e in train]),
              np.deg2rad([e["pip"] for e in train])),
        np.array([e["mass"] for e in train]) / 1000.0 - 0.045])
    y = np.log10([floor_of(e) for e in train])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return coef, float(r2)


def scheduled_thr(coef, ep, mult=3.0):
    x = np.concatenate([
        basis(np.deg2rad(np.array([ep["wave"]])),
              np.deg2rad(np.array([ep["pip"]])))[0],
        [ep["mass"] / 1000.0 - 0.045]])
    return mult * 10.0 ** float(x @ coef)


def adaptive_series(ep, k, block_s=0.1, window_s=2.0, gap_s=0.3,
                    sustain_s=0.05, arm_s=3.5, floor_abs=0.0):
    """Causal running threshold: k x median of trailing block 99th-pcts,
    with a guard gap so an onsetting burst does not raise its own floor,
    firing only if the exceedance is sustained (debounce), and armed only
    after the grasp transients settle (arm_s — a real controller arms the
    detector once grip is established). Returns detection time (nan if
    never fires)."""
    t, e = ep["t"], ep["e"]
    dt = float(np.median(np.diff(t)))
    n_blk = max(1, int(round(block_s / dt)))
    n_sus = max(1, int(round(sustain_s / dt)))
    n_full = len(e) // n_blk
    if n_full < 3:
        return np.nan
    blocks = e[:n_full * n_blk].reshape(n_full, n_blk)
    b99 = np.percentile(blocks, 99, axis=1)
    n_win, n_gap = int(round(window_s / block_s)), int(round(gap_s / block_s))
    i0 = max(n_win + n_gap, int(round(arm_s / block_s)))
    for i in range(i0, n_full):
        floor = np.median(b99[i - n_win - n_gap:i - n_gap])
        thr = max(k * floor, floor_abs)
        for j in np.nonzero(blocks[i] > thr)[0]:
            s = i * n_blk + j
            if np.all(e[s:s + n_sus] > thr) and s + n_sus <= len(e):
                return float(t[s])
    return np.nan


def score_adaptive(eps, k, floor_abs=0.0):
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    lats = []
    for ep in eps:
        td = adaptive_series(ep, k, floor_abs=floor_abs)
        truth = ep["truth"]
        if np.isnan(truth):
            c = "tn" if np.isnan(td) else "fp"
        elif np.isnan(td):
            c = "fn"
        elif td >= truth - 0.3:
            c = "tp"
            lats.append(td - truth)
        else:
            c = "fp"
        counts[c] += 1
    counts["latency_mean"] = round(float(np.mean(lats)), 4) if lats else None
    counts["latency_max"] = round(float(np.max(lats)), 4) if lats else None
    return counts


def main():
    eps = load_episodes()
    # Same deterministic split as eval_slip_big: per class, sorted by
    # (wave, pip, mass, seed), every 3rd trains.
    report = {}
    for rate in ("ideal", "tactip"):
        cfg = DetectorConfig()
        E = energies(eps, cfg, rate)
        pos = sorted([e for e in E if e["kind"] == "pos"],
                     key=lambda e: (e["wave"], e["pip"], e["mass"], e["seed"]))
        neg = sorted([e for e in E if e["kind"] == "neg"],
                     key=lambda e: (e["wave"], e["pip"], e["mass"], e["seed"]))
        train = pos[0::3] + neg[0::3]
        test = [x for i, x in enumerate(pos) if i % 3] + \
               [x for i, x in enumerate(neg) if i % 3]

        # (a) separability: slip-burst peak vs own-episode floor.
        ratios = []
        for ep in [e for e in E if e["kind"] == "pos"
                   and not np.isnan(e["truth"])]:
            sel = ep["t"] >= ep["truth"] - 0.3
            peak = float(ep["e"][sel].max()) if sel.any() else 0.0
            f = floor_of(ep)
            ratios.append(round(peak / f, 2) if f > 0 else np.inf)
        report.setdefault(rate, {})["burst_over_own_floor"] = {
            "min": min(ratios), "median": float(np.median(ratios)),
            "max": max(ratios),
            "n_below_3x": sum(r < 3.0 for r in ratios)}

        # floor spread across conditions (why global calibration fails)
        floors = [floor_of(e) for e in train]
        report[rate]["train_floor_spread"] = {
            "min": round(min(floors), 6), "max": round(max(floors), 6),
            "ratio": round(max(floors) / min(floors), 1)}

        # (b) per-condition calibration.
        thr_c = {}
        for ep in train:
            f = floor_of(ep)
            thr_c[ep["cond"]] = max(thr_c.get(ep["cond"], 0.0), 3.0 * f)
        report[rate]["per_condition"] = score(
            test, lambda ep: thr_c[ep["cond"]])
        report[rate]["per_condition"]["n_test"] = len(test)

        # global 3x-max (reproduces eval_slip_big) for reference
        thr_g = 3.0 * max(floors)
        report[rate]["global_3x_max"] = score(test, lambda ep: thr_g)
        report[rate]["global_3x_max"]["threshold"] = round(thr_g, 6)

        # (c) oracle global threshold sweep on test.
        cands = np.unique([floor_of(e) * k for e in train
                           for k in (1.5, 2, 3, 5, 8, 12, 20)])
        best = None
        for thr in cands:
            s = score(test, lambda ep: thr)
            err = s["fp"] + s["fn"]
            if best is None or err < best[0]:
                best = (err, float(thr), s)
        report[rate]["oracle_global"] = dict(best[2], threshold=round(best[1], 6))

        # (d) scheduled threshold: floor model fitted on training floors.
        fcoef, fr2 = fit_floor_model(train)
        report[rate]["scheduled"] = score(
            test, lambda ep: scheduled_thr(fcoef, ep))
        report[rate]["scheduled"]["floor_fit_r2"] = round(fr2, 3)
        report[rate]["scheduled"]["floor_coef_log10"] = [
            round(float(c), 4) for c in fcoef]

        # (e) adaptive running-floor threshold; k tuned on the train split.
        # The absolute term is a sensor-noise constant (3x the QUIETEST
        # training floor) that stops ratio blow-ups when the local floor
        # is near zero; it needs no per-condition knowledge.
        floor_abs = 3.0 * min(floors)
        best_k = None
        for k in (3.0, 6.0, 12.0, 20.0, 35.0, 60.0, 100.0):
            s = score_adaptive(train, k, floor_abs)
            err = s["fp"] + s["fn"]
            if best_k is None or err < best_k[0] or (err == best_k[0]
                                                     and k > best_k[1]):
                best_k = (err, k)
        report[rate]["adaptive"] = score_adaptive(test, best_k[1], floor_abs)
        report[rate]["adaptive"]["k"] = best_k[1]
        report[rate]["adaptive"]["floor_abs"] = float(floor_abs)

        print(f"[{rate}] floors x{report[rate]['train_floor_spread']['ratio']}"
              f"  burst/floor median {report[rate]['burst_over_own_floor']['median']}"
              f"\n  per-cond:  {report[rate]['per_condition']}"
              f"\n  scheduled: {report[rate]['scheduled']}"
              f"\n  adaptive:  {report[rate]['adaptive']}"
              f"\n  oracle-global: {report[rate]['oracle_global']}", flush=True)

    out = os.path.join(RESULTS, "slip_eval_big_rescored.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
