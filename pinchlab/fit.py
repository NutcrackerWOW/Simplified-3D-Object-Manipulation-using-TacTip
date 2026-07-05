"""Boundary fitting and the paper's equation (plan decision #11).

Physics form, fixed a priori:
    f1*(θ, m) = (m·g / 2) / μ_eff(θ)
where f1* is the minimal stable squeeze per finger and μ_eff is the
*effective* stability ratio — the tangential demand-to-squeeze ratio the
grasp can sustain. μ_eff folds in everything beyond textbook Coulomb
friction: patch curvature, torsional/rolling failure, soft-pad effects.
It is fitted over the posture variables with a quadratic basis:
    μ_eff(wave, pip) = c0 + c1·wave + c2·pip + c3·wave² + c4·pip² + c5·wave·pip
(angles in radians).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .params import G

BASIS_NAMES = ["1", "wave", "pip", "wave^2", "pip^2", "wave*pip"]


def basis(wave_rad: np.ndarray, pip_rad: np.ndarray) -> np.ndarray:
    w, p = np.asarray(wave_rad), np.asarray(pip_rad)
    return np.stack([np.ones_like(w), w, p, w * w, p * p, w * p], axis=-1)


def mu_eff_from_boundary(df_conds: pd.DataFrame) -> pd.DataFrame:
    """μ_eff per condition from the bisection boundary f*."""
    out = df_conds.copy()
    demand = out["mass"] * G / 2.0          # per-finger vertical shear
    out["mu_eff"] = demand / out["f_star"]
    return out


def fit_mu_eff(df: pd.DataFrame):
    """Least-squares fit of μ_eff over the posture basis. Returns (coeffs,
    diagnostics dict)."""
    ok = df.dropna(subset=["f_star", "mu_eff"])
    X = basis(np.deg2rad(ok["wave_deg"]), np.deg2rad(ok["pip_deg"]))
    y = ok["mu_eff"].to_numpy()
    coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return coef, {"r2": r2, "rmse": rmse, "n": len(ok),
                  "mu_min": float(y.min()), "mu_max": float(y.max()),
                  "mu_mean": float(y.mean())}


def logistic_boundary(df_trials: pd.DataFrame, cond_id: str):
    """Per-condition logistic fit P(held | log f): returns (f50, slope)."""
    from sklearn.linear_model import LogisticRegression
    d = df_trials[df_trials["cond_id"] == cond_id]
    d = d[d["outcome"] != "no_grasp"]
    if d["held"].nunique() < 2:
        return np.nan, np.nan
    x = np.log(d["force_setpoint"].to_numpy()).reshape(-1, 1)
    y = d["held"].astype(int).to_numpy()
    m = LogisticRegression(C=1e3).fit(x, y)
    slope = float(m.coef_[0][0])
    if abs(slope) < 1e-9:
        return np.nan, slope
    f50 = float(np.exp(-m.intercept_[0] / slope))
    return f50, slope


def equation_string(coef: np.ndarray) -> str:
    terms = " + ".join(f"({c:+.4f}·{n})" for c, n in zip(coef, BASIS_NAMES))
    return (f"f1*(wave, pip, m) = (m·g/2) / μ_eff,   "
            f"μ_eff = {terms}   [angles in rad]")


def classify_regimes(conds: pd.DataFrame, min_ratio: float = 1.5) -> pd.DataFrame:
    """Tag each posture as friction- or support-regime by mass scaling.

    Coulomb friction predicts f*(2m)/f*(m) = 2. A posture where f* barely
    grows with mass is carrying weight by geometric support (the pads cradle
    the object), so μ_eff = (m·g/2)/f* stops measuring friction there and
    the posture is excluded from the equation fit.
    """
    out = conds.copy()
    out["regime"] = "friction"
    piv = out.pivot_table(index=["wave_deg", "pip_deg"], columns="mass",
                          values="f_star")
    masses = sorted(piv.columns)
    if len(masses) >= 2:
        m0, m1 = masses[0], masses[1]
        expect = m1 / m0
        ratio = piv[m1] / piv[m0]
        support = ratio[ratio < min_ratio * expect / 2.0].index
        mask = out.set_index(["wave_deg", "pip_deg"]).index.isin(support)
        out.loc[mask, "regime"] = "support"
    return out


def fit_report(conds_csv: str, trials_csv: str, out_json: str | None = None):
    conds = pd.read_csv(conds_csv)
    trials = pd.read_csv(trials_csv)
    conds = mu_eff_from_boundary(conds)
    conds = classify_regimes(conds)

    coef_all, diag_all = fit_mu_eff(conds)
    friction = conds[conds["regime"] == "friction"]
    coef, diag = fit_mu_eff(friction)

    # Logistic f50 vs bisection f* consistency
    f50s = []
    for cid in conds["cond_id"]:
        f50, _ = logistic_boundary(trials, cid)
        f50s.append(f50)
    conds["f50_logistic"] = f50s

    # Mass-collapse check: does μ_eff depend on mass? (pure Coulomb → no)
    mass_groups = friction.groupby("mass")["mu_eff"].mean().to_dict()

    support = conds[conds["regime"] == "support"]
    report = {
        "coefficients": {n: float(c) for n, c in zip(BASIS_NAMES, coef)},
        "diagnostics": diag,
        "equation": equation_string(coef),
        "coefficients_global": {n: float(c)
                                for n, c in zip(BASIS_NAMES, coef_all)},
        "diagnostics_global": diag_all,
        "support_regime": {
            "n_conditions": int(len(support)),
            "postures": sorted({(float(w), float(p)) for w, p in
                                zip(support["wave_deg"], support["pip_deg"])}),
            "f_star_range": ([float(support["f_star"].min()),
                              float(support["f_star"].max())]
                             if len(support) else None),
        },
        "mu_eff_by_mass": {str(k): float(v) for k, v in mass_groups.items()},
        "n_conditions": int(len(conds)),
    }
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump(report, fh, indent=2)
    return conds, report


def predict_f_star(coef: np.ndarray, wave_deg: float, pip_deg: float,
                   mass: float) -> float:
    mu = float((basis(np.deg2rad(np.array([wave_deg])),
                      np.deg2rad(np.array([pip_deg]))) @ coef)[0])
    return (mass * G / 2.0) / mu
