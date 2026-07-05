#!/usr/bin/env python3
"""Held-out validation of the fitted equation.

At posture points NOT in the sweep grid, predict f* from the fitted
μ_eff(θ) and test: a grasp at 1.25×f*_pred must hold, one at 0.6×f*_pred
must slip/drop. Reports prediction-interval hit rate.

  python scripts/validate_equation.py --tag sweepB
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.fit import predict_f_star
from pinchlab.params import BoxSpec, TrialSpec, deg
from pinchlab.trial import get_kinematics, run_trial

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

HELD_OUT = [  # (wave°, pip°, mass) — none of these on the sweep grid
    (-11.0, 1.0, 0.05),
    (4.0, 5.0, 0.05),
    (11.0, -3.0, 0.03),
    (-4.0, 7.0, 0.12),
    (9.0, 3.0, 0.09),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", type=str, default="sweepB")
    ap.add_argument("--hi", type=float, default=1.25)
    ap.add_argument("--lo", type=float, default=0.6)
    args = ap.parse_args()

    with open(os.path.join(RESULTS, f"{args.tag}_fit.json")) as fh:
        fit = json.load(fh)
    coef = np.array(list(fit["coefficients"].values()))

    kin = get_kinematics()
    rows = []
    for w, p, m in HELD_OUT:
        posture, _ = kin.solve_mcp_for_gap(deg(w), deg(p), 0.024)
        if posture is None:
            print(f"({w},{p}): infeasible, skipped")
            continue
        f_pred = predict_f_star(coef, w, p, m)
        rec = {"wave": w, "pip": p, "mass": m, "f_pred": f_pred}
        for label, k in (("above", args.hi), ("below", args.lo)):
            spec = TrialSpec(posture=posture, box=BoxSpec(mass=m),
                             force_setpoint=k * f_pred, seed=7,
                             tag=f"eqval_{label}")
            res = run_trial(spec)
            rec[label] = res.outcome
        rec["ok"] = (rec["above"] == "held") and (rec["below"] != "held")
        rows.append(rec)
        print(f"w={w:+6.1f} p={p:+5.1f} m={m*1000:3.0f}g  f*_pred={f_pred:.3f} N  "
              f"@{args.hi:.2f}× → {rec['above']:8s} @{args.lo:.2f}× → "
              f"{rec['below']:8s}  {'✓' if rec['ok'] else '✗'}")

    n_ok = sum(r["ok"] for r in rows)
    print(f"\nprediction-interval hit rate: {n_ok}/{len(rows)}")
    with open(os.path.join(RESULTS, f"{args.tag}_equation_validation.json"), "w") as fh:
        json.dump({"held_out": rows, "hit_rate": f"{n_ok}/{len(rows)}",
                   "hi_factor": args.hi, "lo_factor": args.lo}, fh, indent=2)


if __name__ == "__main__":
    main()