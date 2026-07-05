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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=str, default="all",
                    choices=tuple(LEVELS) + ("all",))
    ap.add_argument("--mass", type=float, default=0.05)
    ap.add_argument("--wave", type=float, default=0.0)
    ap.add_argument("--pip", type=float, default=0.0)
    args = ap.parse_args()

    kin = get_kinematics()
    posture, _ = kin.solve_mcp_for_gap(deg(args.wave), deg(args.pip), 0.024)

    for name in (LEVELS if args.level == "all" else (args.level,)):
        lv = LEVELS[name]
        box = BoxSpec(mass=args.mass, **lv["box"])
        mp = ModelParams(**lv["mp"]) if lv["mp"] else None
        f_star, trials = boundary(posture, box, tag=name, mp=mp)
        mu = (args.mass * G / 2.0) / f_star if np.isfinite(f_star) else np.nan
        pair_mu = ((box.mu_static * 1.2 * 2) / (box.mu_static + 1.2))
        out = {"level": name, "mass": args.mass,
               "box_mu": [box.mu_static, box.mu_dynamic],
               "pair_mu_static": round(pair_mu, 3),
               "tip_modulus_Pa": (lv["mp"].get("hydro_modulus", 5e4)),
               "wave_deg": args.wave, "pip_deg": args.pip,
               "f_star": None if np.isnan(f_star) else round(f_star, 4),
               "mu_eff": None if np.isnan(mu) else round(mu, 4),
               "trials": trials}
        with open(os.path.join(RESULTS, f"material_{name}.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"{name}: f* = {f_star:.3f} N  mu_eff = {mu:.3f}" if
              np.isfinite(f_star) else f"{name}: no stable force found",
              flush=True)


if __name__ == "__main__":
    main()
