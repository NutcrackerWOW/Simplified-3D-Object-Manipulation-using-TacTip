"""Vibration-based slip detector, scored against kinematic ground truth
(plan decisions #9/#10).

Signal: per-finger tangential force magnitude ‖(f2, f3)‖ from the tactile
decomposition. Detector: moving-window band energy of its derivative-like
high-frequency content — a slip event produces a burst of tangential-force
fluctuation as the contact transitions stick→slip.

Two sensor models are evaluated:
  ideal   — every log sample (use --slip-study trials: 1 kHz)
  tactip  — the same signal downsampled to the TacTip camera rate (default
            100 Hz) before detection: what the real optical sensor could see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorConfig:
    highpass_hz: float = 15.0     # remove slow force scheduling/loading trends
    window_s: float = 0.10        # energy integration window
    threshold: float = None       # N²; None → calibrate on training signals
    tactip_hz: float = 100.0


def tangential_signal(series: dict, finger: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(series["t"])
    ft = np.hypot(np.asarray(series[f"f2_{finger}"]),
                  np.asarray(series[f"f3_{finger}"]))
    return t, ft


def downsample(t: np.ndarray, x: np.ndarray, rate_hz: float):
    """Sample-and-hold decimation to the sensor frame rate."""
    if len(t) < 2:
        return t, x
    t_s = np.arange(t[0], t[-1], 1.0 / rate_hz)
    idx = np.searchsorted(t, t_s, side="right") - 1
    return t_s, x[np.clip(idx, 0, len(x) - 1)]


def band_energy(t: np.ndarray, x: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    """High-pass (single-pole) then windowed mean-square energy."""
    if len(t) < 3:
        return np.zeros_like(x)
    dt = float(np.median(np.diff(t)))
    # One-pole high-pass at cfg.highpass_hz (clamped below Nyquist).
    fc = min(cfg.highpass_hz, 0.4 / dt)
    alpha = 1.0 / (1.0 + 2 * np.pi * fc * dt)
    hp = np.zeros_like(x)
    for i in range(1, len(x)):
        hp[i] = alpha * (hp[i - 1] + x[i] - x[i - 1])
    n = max(1, int(round(cfg.window_s / dt)))
    kernel = np.ones(n) / n
    return np.convolve(hp * hp, kernel, mode="same")


def detect(t: np.ndarray, x: np.ndarray, cfg: DetectorConfig,
           threshold: float) -> float:
    """First time the band energy exceeds threshold (nan if never)."""
    e = band_energy(t, x, cfg)
    above = np.nonzero(e > threshold)[0]
    return float(t[above[0]]) if len(above) else np.nan


def score_trials(trials: list, cfg: DetectorConfig, rate: str = "ideal",
                 threshold: float | None = None,
                 pre_tolerance: float = 0.3) -> dict:
    """trials: list of (series, truth_slip_time) — truth nan for clean holds.

    Returns detection stats: threshold used, per-trial detection times,
    true/false positives, and latency stats vs ground truth. A detection up
    to `pre_tolerance` seconds before the kinematic truth counts as a hit
    (anticipatory warning from micro-slip vibration), with negative latency.
    """
    signals = []
    for series, truth in trials:
        t, x = tangential_signal(series, "L")
        _, xr = tangential_signal(series, "R")
        x = np.maximum(x, xr)              # either finger may slip first
        if rate == "tactip":
            t, x = downsample(t, x, cfg.tactip_hz)
        signals.append((t, x, truth))

    if threshold is None:
        # Calibrate: highest noise floor among pre-slip/clean segments ×3.
        floors = []
        for t, x, truth in signals:
            e = band_energy(t, x, cfg)
            t_end = truth if not np.isnan(truth) else t[-1]
            pre = e[t < t_end - 0.05]
            if len(pre):
                floors.append(np.percentile(pre, 99))
        threshold = 3.0 * max(floors) if floors else 1e-6

    det_times, latencies = [], []
    tp = fp = fn = tn = 0
    for t, x, truth in signals:
        td = detect(t, x, cfg, threshold)
        det_times.append(td)
        if np.isnan(truth):
            if np.isnan(td):
                tn += 1
            else:
                fp += 1
        else:
            if np.isnan(td):
                fn += 1
            elif td >= truth - pre_tolerance:
                tp += 1
                latencies.append(td - truth)
            else:                          # fired long before physical slip
                fp += 1
    return {
        "rate": rate, "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n": len(signals),
        "latency_mean": float(np.mean(latencies)) if latencies else np.nan,
        "latency_max": float(np.max(latencies)) if latencies else np.nan,
        "det_times": det_times,
    }
