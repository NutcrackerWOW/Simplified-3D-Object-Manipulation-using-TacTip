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

MASS_REF = 0.045   # kg — mass term centered mid-range so c0 keeps meaning
BASIS_NAMES = ["1", "wave", "pip", "wave^2", "pip^2", "wave*pip",
               f"(m-{MASS_REF})"]


def basis(wave_rad: np.ndarray, pip_rad: np.ndarray) -> np.ndarray:
    """Posture-only quadratic basis (6 columns)."""
    w, p = np.asarray(wave_rad), np.asarray(pip_rad)
    return np.stack([np.ones_like(w), w, p, w * w, p * p, w * p], axis=-1)


def mu_eff_model(coef: np.ndarray, wave_deg, pip_deg, mass) -> np.ndarray:
    """Evaluate fitted μ_eff(wave, pip, m). Accepts 6-coef (posture-only,
    legacy) or 7-coef (posture + linear mass) fits."""
    b = basis(np.deg2rad(np.asarray(wave_deg, dtype=float)),
              np.deg2rad(np.asarray(pip_deg, dtype=float)))
    mu = b @ np.asarray(coef)[:6]
    if len(coef) > 6:
        mu = mu + coef[6] * (np.asarray(mass, dtype=float) - MASS_REF)
    return mu


def mu_eff_from_boundary(df_conds: pd.DataFrame) -> pd.DataFrame:
    """μ_eff per condition from the bisection boundary f*."""
    out = df_conds.copy()
    demand = out["mass"] * G / 2.0          # per-finger vertical shear
    out["mu_eff"] = demand / out["f_star"]
    return out


def fit_mu_eff(df: pd.DataFrame, sigma_rel_by_mass: dict | None = None):
    """Least-squares fit of μ_eff over the posture basis. Returns (coeffs,
    diagnostics dict).

    If sigma_rel_by_mass is given (relative seed-noise σ of f* per mass,
    measured by the sweep audit), the fit is weighted 1/σ² so the noisier
    high-mass boundaries don't dominate; R²/RMSE stay unweighted for
    comparability."""
    ok = df.dropna(subset=["f_star", "mu_eff"])
    X = np.column_stack([
        basis(np.deg2rad(ok["wave_deg"]), np.deg2rad(ok["pip_deg"])),
        ok["mass"].to_numpy() - MASS_REF])
    y = ok["mu_eff"].to_numpy()
    if sigma_rel_by_mass:
        fallback = max(sigma_rel_by_mass.values())
        sig = np.array([sigma_rel_by_mass.get(round(m, 3), fallback)
                        for m in ok["mass"]]) * y
        wgt = 1.0 / np.maximum(sig, 1e-9)
        coef, res, rank, sv = np.linalg.lstsq(
            X * wgt[:, None], y * wgt, rcond=None)
    else:
        coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return coef, {"r2": r2, "rmse": rmse, "n": len(ok),
                  "weighted": bool(sigma_rel_by_mass),
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


def seed_sigma_from_audit(audit_json: str) -> dict:
    """Per-mass relative seed noise of f* (median std/median over the audit's
    friction-regime conditions, pip ≥ 0)."""
    with open(audit_json) as fh:
        audit = json.load(fh)
    by_mass = {}
    for key, s in audit["summary"].items():
        pip = float(key.split("_p")[1].split("_")[0])
        mass = round(float(key.split("_m")[1]) / 1000.0, 3)
        if pip >= 0 and s.get("n_valid", 0) >= 3:
            by_mass.setdefault(mass, []).append(s["std_rel"])
    return {m: float(np.median(v)) for m, v in by_mass.items()}


def fit_report(conds_csv: str, trials_csv: str, out_json: str | None = None,
               fragile_pips: tuple = (-5.0,), audit_json: str | None = None):
    """fragile_pips: postures the seed audit showed to be bimodal (boundary
    existence is seed-dependent — at pip=−5° only 2/5 independent seeds find
    a stable anchor, and those land at μ_eff above the Coulomb pair). They
    are excluded from the friction fit and reported separately."""
    conds = pd.read_csv(conds_csv)
    trials = pd.read_csv(trials_csv)
    conds = mu_eff_from_boundary(conds)
    conds = classify_regimes(conds)
    conds.loc[(conds["regime"] == "friction")
              & conds["pip_deg"].isin(fragile_pips), "regime"] = "fragile"

    if audit_json is None:
        default = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "results", "sweep_audit.json")
        audit_json = default if os.path.exists(default) else None
    sigma_rel = seed_sigma_from_audit(audit_json) if audit_json else None

    coef_all, diag_all = fit_mu_eff(conds)
    friction = conds[conds["regime"] == "friction"]
    coef, diag = fit_mu_eff(friction, sigma_rel_by_mass=sigma_rel)

    # Logistic f50 vs bisection f* consistency
    f50s = []
    for cid in conds["cond_id"]:
        f50, _ = logistic_boundary(trials, cid)
        f50s.append(f50)
    conds["f50_logistic"] = f50s

    # Mass-collapse check: does μ_eff depend on mass? (pure Coulomb → no)
    mass_groups = friction.groupby("mass")["mu_eff"].mean().to_dict()

    support = conds[conds["regime"] == "support"]
    fragile = conds[conds["regime"] == "fragile"]
    report = {
        "coefficients": {n: float(c) for n, c in zip(BASIS_NAMES, coef)},
        "diagnostics": diag,
        "equation": equation_string(coef),
        "seed_noise_sigma_rel": ({str(k): v for k, v in sigma_rel.items()}
                                 if sigma_rel else None),
        "fragile_regime": {
            "pips_deg": list(fragile_pips),
            "n_conditions": int(len(fragile)),
            "reason": "audit: boundary existence seed-dependent (bimodal)",
        },
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
    mu = float(mu_eff_model(coef, np.array([wave_deg]),
                            np.array([pip_deg]), mass)[0])
    return (mass * G / 2.0) / mu
