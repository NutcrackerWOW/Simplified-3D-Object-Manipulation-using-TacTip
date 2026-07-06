#!/usr/bin/env python3
"""M8b: material study — the mechanism-discriminating experiment.

If the boundary were Coulomb-governed, f* ∝ 1/μ (halving friction doubles the
needed squeeze). If it is rolling-governed, f* should be largely insensitive
to the friction pair but sensitive to the fingertip's hydroelastic modulus
(softer pad → larger contact patch → more rolling resistance).

Levels (box, 50 g, reference posture wave 0 / PIP 0):
  friction  low  μs/μd 0.4/0.3   base 1.0/0.8   high 1.5/1.2
  tip       soft E 25 kPa        base 50 kPa    stiff E 100 kPa

  python scripts/eval_materials.py --level fric_low     # one level
  python scripts/eval_materials.py --level all          # sequential
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from eval_shapes import boundary
from pinchlab.params import G, BoxSpec, ModelParams, deg
from pinchlab.trial import get_kinematics

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

LEVELS = {
    # box friction variants (tip material at baseline)
    "fric_low":  {"box": dict(mu_static=0.4, mu_dynamic=0.3), "mp": {}},
    "fric_base": {"box": dict(mu_static=1.0, mu_dynamic=0.8), "mp": {}},
    "fric_high": {"box": dict(mu_static=1.5, mu_dynamic=1.2), "mp": {}},
    # tip stiffness variants (box friction at baseline)
    "tip_soft":  {"box": {}, "mp": dict(hydro_modulus=2.5e4)},
    "tip_stiff": {"box": {}, "mp": dict(hydro_modulus=1.0e5)},
}


def run_level(job):
    """One material row (picklable job dict — usable from a spawn Pool)."""
    name = job["level"]
    lv = LEVELS[name]
    kin = get_kinematics()
    posture, _ = kin.solve_mcp_for_gap(
        deg(job["wave"]), deg(job["pip"]), 0.024)
    box = BoxSpec(mass=job["mass"], **lv["box"])
    mp = ModelParams(**lv["mp"]) if lv["mp"] else None
    f_star, trials = boundary(posture, box, n_bisect=job["n_bisect"],
                              tag=name, mp=mp,
                              seeds=tuple(range(job["seeds"])))
    mu = (job["mass"] * G / 2.0) / f_star if np.isfinite(f_star) else np.nan
    pair_mu = ((box.mu_static * 1.2 * 2) / (box.mu_static + 1.2))
    out = {"level": name, "mass": job["mass"],
           "box_mu": [box.mu_static, box.mu_dynamic],
           "pair_mu_static": round(pair_mu, 3),
           "tip_modulus_Pa": (lv["mp"].get("hydro_modulus", 5e4)),
           "wave_deg": job["wave"], "pip_deg": job["pip"],
           "n_seeds": job["seeds"], "n_bisect": job["n_bisect"],
           "f_star": None if np.isnan(f_star) else round(f_star, 4),
           "mu_eff": None if np.isnan(mu) else round(mu, 4),
           "trials": trials}
    with open(os.path.join(RESULTS, f"material_{name}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"{name}: f* = {f_star:.3f} N  mu_eff = {mu:.3f}" if
          np.isfinite(f_star) else f"{name}: no stable force found",
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=str, default="all",
                    choices=tuple(LEVELS) + ("all",))
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--wave", type=float, default=0.0)
    ap.add_argument("--pip", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n-bisect", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallelize across levels (spawn pool)")
    args = ap.parse_args()

    levels = tuple(LEVELS) if args.level == "all" else (args.level,)
    jobs = [dict(level=n, mass=args.mass, wave=args.wave, pip=args.pip,
                 seeds=args.seeds, n_bisect=args.n_bisect) for n in levels]
    if args.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp_
        ctx = mp_.get_context("spawn")
        with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
            for _ in pool.imap_unordered(run_level, jobs):
                pass
    else:
        for job in jobs:
            run_level(job)


if __name__ == "__main__":
    main()
