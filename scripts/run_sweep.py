#!/usr/bin/env python3
"""Run the Phase A minimal-force sweep (headless, parallel, resumable).

Examples
--------
  python scripts/run_sweep.py                    # full configured sweep
  python scripts/run_sweep.py --smoke            # 2 conditions, 1 repeat
  python scripts/run_sweep.py --workers 8 --tag sweepA
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinchlab.sweep import SweepConfig, run_sweep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--tag", type=str, default="sweep")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--bisect", type=int, default=7)
    ap.add_argument("--masses", type=float, nargs="+", default=[0.03, 0.06, 0.12])
    ap.add_argument("--waves", type=float, nargs="+",
                    default=[-15.0, -7.5, 0.0, 7.5, 15.0])
    ap.add_argument("--pips", type=float, nargs="+",
                    default=[-5.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sweep to validate the machinery")
    args = ap.parse_args()

    cfg = SweepConfig(
        waves_deg=tuple(args.waves), pips_deg=tuple(args.pips),
        masses=tuple(args.masses), workers=args.workers,
        n_repeat=args.repeats, n_bisect=args.bisect, tag=args.tag)
    if args.smoke:
        cfg = SweepConfig(waves_deg=(0.0,), pips_deg=(0.0, 4.0), masses=(0.05,),
                          n_repeat=1, n_bisect=3, workers=2, tag="smoke")

    df = run_sweep(cfg)
    print("\ncondition summaries:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
