#!/usr/bin/env python3
"""Compare the reduced (one-liner) grip-force rule against the full quadratic
fit over the 16 held-out validation conditions.

The held-out validation (results/sweepB_equation_validation.json) probed each
condition at 1.25x and 0.6x the FULL quadratic model's predicted f*.  The paper
headlines the reduced rule

    mu_eff = 0.71 - 4.2 (m - 0.045)      [m in kg]

so this script checks that the reduced rule's f* predictions stay inside those
experimentally probed brackets, i.e. that the 16/16 hold/drop validation
transfers to the reduced rule.  Writes results/one_liner_vs_quadratic.json.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = 9.81  # matches pinchlab.params.G

# exact WLS coefficients (results/sweepB_fit.json): intercept + mass term
CORE_C0, CORE_CM = 0.7069192246342931, -4.195764821992556
# rounded form as printed in the paper
PAPER_C0, PAPER_CM = 0.71, -4.2
MASS_REF = 0.045


def f_star(mu, mass):
    return (mass * G / 2.0) / mu


def main():
    val = json.loads((ROOT / "results/sweepB_equation_validation.json").read_text())
    rows = []
    for p in val["in_envelope"]:
        m, f_quad = p["mass"], p["f_pred"]
        f_paper = f_star(PAPER_C0 + PAPER_CM * (m - MASS_REF), m)
        f_core = f_star(CORE_C0 + CORE_CM * (m - MASS_REF), m)
        rows.append({
            "wave": p["wave"], "pip": p["pip"], "mass": m,
            "f_quad": round(f_quad, 4),
            "f_one_liner": round(f_paper, 4),
            "f_core_exact": round(f_core, 4),
            "ratio_one_liner": round(f_paper / f_quad, 4),
            "in_probe_bracket": 0.6 <= f_paper / f_quad <= 1.25,
        })
    ratios = [r["ratio_one_liner"] for r in rows]
    out = {
        "description": "reduced-rule f* vs full-quadratic f* over the 16 "
                       "held-out validation conditions",
        "one_liner": f"mu_eff = {PAPER_C0} {PAPER_CM:+g}*(m-{MASS_REF})",
        "n": len(rows),
        "ratio_min": min(ratios), "ratio_max": max(ratios),
        "all_within_probe_brackets": all(r["in_probe_bracket"] for r in rows),
        "rows": rows,
    }
    dest = ROOT / "results/one_liner_vs_quadratic.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"n={out['n']}  ratio one-liner/quadratic: "
          f"{out['ratio_min']:.3f}..{out['ratio_max']:.3f}  "
          f"all within 0.6x-1.25x brackets: {out['all_within_probe_brackets']}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
