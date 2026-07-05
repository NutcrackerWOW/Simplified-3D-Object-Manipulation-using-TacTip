#!/usr/bin/env python3
"""M5: fit the stability boundary map and export the paper's equation.

  python scripts/fit_map.py --tag sweepA
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from pinchlab.fit import fit_report, mu_eff_from_boundary
from pinchlab import plots

RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", type=str, default="sweepA")
    args = ap.parse_args()

    conds_csv = os.path.join(RESULTS, f"{args.tag}_conditions.csv")
    trials_csv = os.path.join(RESULTS, f"{args.tag}_trials.csv")
    out_json = os.path.join(RESULTS, f"{args.tag}_fit.json")

    conds, report = fit_report(conds_csv, trials_csv, out_json)

    print(f"n_conditions = {report['n_conditions']}")
    sup = report["support_regime"]
    print(f"support-regime postures (mass-independent f*, excluded from fit): "
          f"{sup['postures']}  f* range {sup['f_star_range']}")
    print(f"μ_eff (friction regime): mean {report['diagnostics']['mu_mean']:.3f}  "
          f"range [{report['diagnostics']['mu_min']:.3f}, "
          f"{report['diagnostics']['mu_max']:.3f}]")
    print(f"friction-regime fit R² = {report['diagnostics']['r2']:.3f}  "
          f"RMSE = {report['diagnostics']['rmse']:.3f}   "
          f"(global incl. support rows: R² = "
          f"{report['diagnostics_global']['r2']:.3f})")
    print("μ_eff by mass:", {k: round(v, 3)
                             for k, v in report['mu_eff_by_mass'].items()})
    print("\nEQUATION:\n ", report["equation"])

    plots.plot_map(conds, os.path.join(RESULTS, f"{args.tag}_mu_eff_map.png"),
                   value="mu_eff", label="μ_eff = (m·g/2) / f*")
    plots.plot_map(conds, os.path.join(RESULTS, f"{args.tag}_fstar_map.png"),
                   value="f_star", label="minimal stable squeeze f* (N)")

    # Example boundary figure: median-f* condition
    trials = pd.read_csv(trials_csv)
    ok = conds.dropna(subset=["f_star"])
    row = ok.iloc[(ok["f_star"] - ok["f_star"].median()).abs().argsort().iloc[0]]
    plots.plot_boundary(trials, row,
                        os.path.join(RESULTS, f"{args.tag}_boundary_example.png"))
    print(f"\nfigures → results/{args.tag}_*.png, report → {out_json}")


if __name__ == "__main__":
    main()
