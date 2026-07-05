"""Phase A sweep: minimal stable force across the grasp manifold.

Conditions = wave-tilt × PIP grid (MCP solved per point to land the pads on
the box) × object masses. Per condition, the minimal stable squeeze f1* is
found by log-space bisection; a probe force is "stable" only if EVERY seed
repeat holds. Every probe trial's summary row is persisted (that's the ML
dataset); per-condition boundary results go to a second file. Both files are
CSV and the sweep is resumable (finished condition ids are skipped).

Workers use the multiprocessing "spawn" context — Drake objects must never
cross a fork boundary.
"""

from __future__ import annotations

import multiprocessing as mp_
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .params import BoxSpec, ControlParams, Posture, TrialSpec, deg
from .trial import get_kinematics, run_trial

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


@dataclass
class SweepConfig:
    waves_deg: tuple = (-15.0, -7.5, 0.0, 7.5, 15.0)
    pips_deg: tuple = (-5.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
    masses: tuple = (0.03, 0.06, 0.12)
    box: BoxSpec = field(default_factory=BoxSpec)
    # 2.5 N ceiling: ≥5 N squeezes the box off the dome centers and the
    # grasp degenerates (measured) — boundaries live at 0.2–1.5 N here.
    f_lo: float = 0.05
    f_hi: float = 2.5
    n_bisect: int = 7
    n_repeat: int = 2
    workers: int = 14
    out_dir: str = RESULTS_DIR
    tag: str = "sweep"


@dataclass
class Condition:
    cond_id: str
    wave_deg: float
    pip_deg: float
    mcp_deg: float          # solved
    mass: float
    posture: Posture
    gap: float
    tilt_deg: float


def generate_conditions(cfg: SweepConfig) -> list:
    """Solve the manifold; only feasible grasp postures become conditions."""
    kin = get_kinematics()
    conds, skipped = [], []
    for w in cfg.waves_deg:
        for p in cfg.pips_deg:
            posture, g = kin.solve_mcp_for_gap(
                deg(w), deg(p), cfg.box.size - 0.001)
            if posture is None:
                skipped.append((w, p, "unsolvable"))
                continue
            feas = kin.grasp_feasibility(posture, cfg.box.size)
            if not feas["feasible"]:
                skipped.append((w, p, "filtered"))
                continue
            for m in cfg.masses:
                cid = f"w{w:+05.1f}_p{p:+05.1f}_m{int(round(m*1000)):03d}"
                conds.append(Condition(
                    cid, w, p, float(np.rad2deg(posture.mcp_l)), m,
                    posture, feas["gap"], feas["tilt_deg"]))
    return conds, skipped


def _probe(cond: Condition, cfg: SweepConfig, force: float, probe_idx: int):
    """Run all repeats at one force level; returns (stable, rows)."""
    rows, stable = [], True
    for rep in range(cfg.n_repeat):
        spec = TrialSpec(
            posture=cond.posture,
            box=replace(cfg.box, mass=cond.mass),
            force_setpoint=float(force),
            seed=1000 * probe_idx + rep,
            tag=cfg.tag,
        )
        res = run_trial(spec)
        row = res.flat_row()
        row.update(cond_id=cond.cond_id, wave_deg=cond.wave_deg,
                   pip_deg=cond.pip_deg, mcp_solved_deg=cond.mcp_deg,
                   probe_idx=probe_idx, repeat=rep)
        rows.append(row)
        if not res.held:
            stable = False
    return stable, rows


def run_condition(args):
    """Worker: bisection for one condition. Returns (summary_row, trial_rows)."""
    cond, cfg = args
    t_start = time.time()
    all_rows = []
    probe_idx = 0

    # Stability is a WINDOW in force: too little → gravity slip, too much →
    # the box is squeezed out of the dome grip (measured at ≥2.5 N). The
    # anchor must land inside the window, so climb a mass-scaled ladder
    # (~3/6/12 × the naive Coulomb boundary) and bisect below the first
    # stable rung.
    W = cond.mass * 9.81 / 2.0
    ladder = sorted({float(np.clip(3.0 * W * k, 0.3, cfg.f_hi))
                     for k in (1.0, 2.0, 4.0)})
    anchor = None
    for fa in ladder:
        stable, rows = _probe(cond, cfg, fa, probe_idx)
        all_rows += rows
        probe_idx += 1
        if stable:
            anchor = fa
            break
    f_star = np.nan
    if anchor is not None:
        hi, lo = anchor, cfg.f_lo
        # Optional cheap check: maybe even f_lo holds.
        stable_lo, rows = _probe(cond, cfg, lo, probe_idx)
        all_rows += rows
        probe_idx += 1
        if stable_lo:
            f_star = lo
        else:
            for _ in range(cfg.n_bisect):
                mid = float(np.sqrt(lo * hi))
                stable, rows = _probe(cond, cfg, mid, probe_idx)
                all_rows += rows
                probe_idx += 1
                if stable:
                    hi = mid
                else:
                    lo = mid
            f_star = hi

    n_no_grasp = sum(r["outcome"] == "no_grasp" for r in all_rows)
    summary = {
        "cond_id": cond.cond_id, "wave_deg": cond.wave_deg,
        "pip_deg": cond.pip_deg, "mcp_solved_deg": cond.mcp_deg,
        "mass": cond.mass, "gap": cond.gap, "tilt_deg": cond.tilt_deg,
        "f_star": f_star, "n_trials": len(all_rows),
        "n_no_grasp": n_no_grasp, "wall_s": time.time() - t_start,
    }
    return summary, all_rows


def run_sweep(cfg: SweepConfig, progress=print) -> pd.DataFrame:
    os.makedirs(cfg.out_dir, exist_ok=True)
    trials_path = os.path.join(cfg.out_dir, f"{cfg.tag}_trials.csv")
    conds_path = os.path.join(cfg.out_dir, f"{cfg.tag}_conditions.csv")

    conds, skipped = generate_conditions(cfg)
    progress(f"{len(conds)} conditions ({len(skipped)} grid points skipped: "
             f"{[s[:2] for s in skipped][:10]}{'…' if len(skipped) > 10 else ''})")

    done = set()
    if os.path.exists(conds_path):
        done = set(pd.read_csv(conds_path)["cond_id"])
        progress(f"resuming: {len(done)} conditions already complete")
    todo = [c for c in conds if c.cond_id not in done]
    if not todo:
        progress("nothing to do")
        return pd.read_csv(conds_path)

    ctx = mp_.get_context("spawn")
    t0 = time.time()
    n_done = 0
    with ctx.Pool(processes=min(cfg.workers, len(todo))) as pool:
        for summary, rows in pool.imap_unordered(
                run_condition, [(c, cfg) for c in todo]):
            _append_csv(conds_path, [summary])
            _append_csv(trials_path, rows)
            n_done += 1
            rate = (time.time() - t0) / n_done
            eta = rate * (len(todo) - n_done) / 60
            progress(f"[{n_done}/{len(todo)}] {summary['cond_id']} "
                     f"f*={summary['f_star']:.3f} N "
                     f"({summary['n_trials']} trials, {summary['wall_s']:.0f}s) "
                     f"ETA {eta:.0f} min")
    return pd.read_csv(conds_path)


def _append_csv(path: str, rows: list) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
