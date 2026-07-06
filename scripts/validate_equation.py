#!/usr/bin/env python3
"""Held-out validation of the fitted equation.

At posture/mass points NOT in the sweep grid, predict f* from the fitted
μ_eff(wave, pip, m) and test: a grasp at 1.25×f*_pred must hold, one at
0.6×f*_pred must slip/drop — each probed with 2 spawn seeds (all seeds must
agree). 16 points lie inside the fitted envelope (friction regime,
pip 0.5–9.5°, wave ±13°, 30–60 g); 3 deliberate out-of-envelope probes
demonstrate the characterized limits (fragile pip band, 120 g mass ceiling,
90 g mass extrapolation).

  python scripts/validate_equation.py --tag sweepB --workers 10
"""

import argparse
import json
import multiprocessing as mp_
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pinchlab.fit import predict_f_star
from pinchlab.params import BoxSpec, TrialSpec, deg
from pinchlab.trial import get_kinematics, run_trial

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")

# (wave°, pip°, mass) — none on the sweep grid (waves ±15/±7.5/0,
# pips −5…10 in 2° steps, masses 30/60/120 g).
IN_ENVELOPE = [
    (-12.0, 1.5, 0.040), (-9.0, 3.5, 0.055), (-5.0, 6.5, 0.030),
    (-2.0, 8.5, 0.060), (2.0, 0.5, 0.045), (5.0, 2.5, 0.060),
    (9.0, 5.5, 0.035), (12.0, 7.5, 0.050), (-13.0, 4.5, 0.060),
    (13.0, 2.5, 0.030), (-3.0, 5.5, 0.050), (6.0, 8.5, 0.040),
    (10.0, 0.5, 0.060), (-6.0, 2.5, 0.035), (0.5, 3.5, 0.055),
    (3.5, 6.5, 0.045),
]
OUT_OF_ENVELOPE = [   # each violates a characterized limit on purpose
    (11.0, -3.0, 0.030),   # fragile/support pip band (excluded from fit)
    (-4.0, 7.0, 0.120),    # 120 g: stable window closed
    (9.0, 3.0, 0.090),     # mass extrapolation beyond fitted 30-60 g
]


def run_probe(job):
    w, p, m, label, force, seed = job
    kin = get_kinematics()
    posture, _ = kin.solve_mcp_for_gap(deg(w), deg(p), 0.024)
    if posture is None:
        return dict(wave=w, pip=p, mass=m, label=label, seed=seed,
                    outcome="infeasible")
    spec = TrialSpec(posture=posture, box=BoxSpec(mass=m),
                     force_setpoint=force, seed=seed,
                     tag=f"eqval_{label}")
    res = run_trial(spec)
    print(f"w={w:+6.1f} p={p:+5.1f} m={m*1000:3.0f}g {label:5s} "
          f"f={force:.3f} seed={seed}: {res.outcome}", flush=True)
    return dict(wave=w, pip=p, mass=m, label=label, seed=seed,
                force=round(force, 4), outcome=res.outcome,
                held=bool(res.held))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", type=str, default="sweepB")
    ap.add_argument("--hi", type=float, default=1.25)
    ap.add_argument("--lo", type=float, default=0.6)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    with open(os.path.join(RESULTS, f"{args.tag}_fit.json")) as fh:
        fit = json.load(fh)
    coef = np.array(list(fit["coefficients"].values()))

    jobs = []
    for pts, env in ((IN_ENVELOPE, "in"), (OUT_OF_ENVELOPE, "out")):
        for (w, p, m) in pts:
            f_pred = predict_f_star(coef, w, p, m)
            for label, k in (("above", args.hi), ("below", args.lo)):
                for s in range(args.seeds):
                    jobs.append((w, p, m, label, k * f_pred, 100 + s))
    print(f"{len(jobs)} probe trials "
          f"({len(IN_ENVELOPE)} in-envelope + {len(OUT_OF_ENVELOPE)} "
          f"out-of-envelope points x 2 probes x {args.seeds} seeds)",
          flush=True)

    ctx = mp_.get_context("spawn")
    with ctx.Pool(processes=min(args.workers, len(jobs))) as pool:
        rows = list(pool.imap_unordered(run_probe, jobs))

    def summarize(points, env):
        out = []
        for (w, p, m) in points:
            rr = [r for r in rows
                  if (r["wave"], r["pip"], r["mass"]) == (w, p, m)]
            above = [r for r in rr if r["label"] == "above"]
            below = [r for r in rr if r["label"] == "below"]
            rec = {"wave": w, "pip": p, "mass": m,
                   "f_pred": round(predict_f_star(coef, w, p, m), 4),
                   "above": [r["outcome"] for r in above],
                   "below": [r["outcome"] for r in below],
                   "above_ok": all(r.get("held") for r in above),
                   "below_ok": all(not r.get("held", True) for r in below)}
            rec["ok"] = rec["above_ok"] and rec["below_ok"]
            out.append(rec)
            print(f"[{env}] w={w:+6.1f} p={p:+5.1f} m={m*1000:3.0f}g "
                  f"f*={rec['f_pred']:.3f}  above={rec['above']} "
                  f"below={rec['below']}  {'OK' if rec['ok'] else 'MISS'}",
                  flush=True)
        return out

    in_rows = summarize(IN_ENVELOPE, "in")
    out_rows = summarize(OUT_OF_ENVELOPE, "out")
    n_ok = sum(r["ok"] for r in in_rows)
    n_above = sum(r["above_ok"] for r in in_rows)
    n_below = sum(r["below_ok"] for r in in_rows)
    print(f"\nin-envelope: {n_ok}/{len(in_rows)} full hits "
          f"(hold@{args.hi}x: {n_above}/{len(in_rows)}, "
          f"drop@{args.lo}x: {n_below}/{len(in_rows)})")

    with open(os.path.join(RESULTS,
                           f"{args.tag}_equation_validation.json"), "w") as fh:
        json.dump({"in_envelope": in_rows, "out_of_envelope": out_rows,
                   "hit_rate": f"{n_ok}/{len(in_rows)}",
                   "hold_rate": f"{n_above}/{len(in_rows)}",
                   "drop_rate": f"{n_below}/{len(in_rows)}",
                   "hi_factor": args.hi, "lo_factor": args.lo,
                   "seeds": args.seeds}, fh, indent=2)


if __name__ == "__main__":
    main()
