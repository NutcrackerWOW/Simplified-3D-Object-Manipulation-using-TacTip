#!/usr/bin/env python3
"""Sweep variance audit: is the sweep's f* limited by seed noise or by the
bisection notch?

For 12 representative conditions (both regimes, all fitted masses), run 5
INDEPENDENT single-seed bisections (same anchor-ladder + log-bisection scheme
as the sweep, n_bisect=7) and compare the seed-induced spread of f* against
the bisection notch width. Each (condition, seed) pair is one worker job.

  python scripts/run_audit.py --workers 10
"""

import argparse
import json
import multiprocessing as mp_
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from eval_shapes import boundary
from pinchlab.params import BoxSpec, G, deg
from pinchlab.trial import get_kinematics

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

# (wave_deg, pip_deg, mass_kg) — friction regime spread + support regime
# (0,-2) + steep-boundary pip=-5 rows + both fitted masses.
CONDITIONS = [
    (0.0, 0.0, 0.03), (0.0, 0.0, 0.06),
    (7.5, 4.0, 0.03), (7.5, 4.0, 0.06),
    (-7.5, 4.0, 0.06), (-15.0, 0.0, 0.03),
    (15.0, 8.0, 0.03), (15.0, 8.0, 0.06),
    (0.0, 8.0, 0.06), (0.0, -5.0, 0.03),
    (0.0, -5.0, 0.06), (0.0, -2.0, 0.06),
]


def run_one(job):
    w, p, m, seed = job
    kin = get_kinematics()
    posture, _ = kin.solve_mcp_for_gap(deg(w), deg(p), 0.024)
    f_star, trials = boundary(
        posture, BoxSpec(mass=m), n_bisect=7,
        tag=f"audit_w{w:+.1f}_p{p:+.1f}_m{m*1000:.0f}_s{seed}",
        seeds=(seed,))
    print(f"w={w:+5.1f} p={p:+4.1f} m={m*1000:.0f}g seed={seed}: "
          f"f*={f_star:.4f}" if np.isfinite(f_star) else
          f"w={w:+5.1f} p={p:+4.1f} m={m*1000:.0f}g seed={seed}: no boundary",
          flush=True)
    return dict(wave_deg=w, pip_deg=p, mass=m, seed=seed,
                f_star=None if np.isnan(f_star) else round(f_star, 4),
                n_trials=len(trials))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    jobs = [(w, p, m, s) for (w, p, m) in CONDITIONS
            for s in range(args.seeds)]
    print(f"{len(jobs)} independent bisections "
          f"({len(CONDITIONS)} conditions x {args.seeds} seeds), "
          f"{args.workers} workers", flush=True)

    ctx = mp_.get_context("spawn")
    with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
        rows = list(pool.imap_unordered(run_one, jobs))

    summary = {}
    for (w, p, m) in CONDITIONS:
        fs = [r["f_star"] for r in rows
              if (r["wave_deg"], r["pip_deg"], r["mass"]) == (w, p, m)
              and r["f_star"] is not None]
        key = f"w{w:+.1f}_p{p:+.1f}_m{m*1000:.0f}"
        if not fs:
            summary[key] = {"n_valid": 0}
            continue
        fs = sorted(fs)
        med = float(np.median(fs))
        # Relative seed spread vs the local notch width (one bisection step
        # around the median: factor exp(ln(anchor/f_lo)/2^7) is condition-
        # dependent; report the empirical spread instead).
        summary[key] = {
            "n_valid": len(fs), "f_star_all": fs,
            "median": round(med, 4),
            "spread_rel": round((fs[-1] - fs[0]) / med, 4),
            "std_rel": round(float(np.std(fs)) / med, 4),
            "mu_eff_median": round((m * G / 2.0) / med, 4),
        }
        print(f"{key}: f* {fs}  spread {summary[key]['spread_rel']*100:.1f}% "
              f"of median", flush=True)

    with open(os.path.join(RESULTS, "sweep_audit.json"), "w") as fh:
        json.dump({"seeds": args.seeds, "n_bisect": 7,
                   "summary": summary, "runs": rows}, fh, indent=2)
    print("→ results/sweep_audit.json", flush=True)


if __name__ == "__main__":
    main()
