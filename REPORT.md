# 3D Object Manipulation, Pinching and Slipping Using Tactile Sensors
### Simulation study report — draft for paper writing

**System:** two-finger "Touch_Finger" hand, TacTip-style soft fingertips, Drake 1.54
(SAP discrete solver, compliant-hydroelastic contact).
**Deliverables:** a closed-form minimal-grip-force equation with a measured validity
envelope; a slip detector evaluated at the real sensor's camera rate; moving-grasp
validation; a shape study isolating the failure mechanism.

---

## 1. Method summary

- **Hand model.** `model/Touch_Finger/Complete.urdf`, fingers pointing down, pinch
  axis horizontal so gravity loads the contacts in shear. Wave/MCP/PIP actively
  actuated (posture PD + gravity feedforward); DIP passive (torsional spring toward
  the measured PIP angle, k = 0.35 N·m/rad, ≈5° give at 1–2 N). Fingertips are
  compliant hydroelastic (E = 50 kPa, Hunt–Crossley dissipation 1.5, μs/μd = 1.2/0.9);
  objects rigid hydroelastic (μs/μd = 1.0/0.8) → pair Coulomb μ ≈ 1.09.
- **Tactile signal.** Per fingertip, the hydroelastic contact resultant force and
  patch centroid stand in for the TacTip (idealization; sensor realism is covered by a
  100 Hz camera-rate variant). Grasp frame from the two centroids: e1 = grasp axis
  (squeeze f1), e2 = vertical shear (weight-bearing f2), e3 = lateral (f3);
  utilization ρ = ‖(f2,f3)‖ / f1.
- **Control.** State machine (close → grip → release support → hold). An outer loop
  servos the *squeeze component* f1 to a setpoint via integral torque trim on MCP.
- **Grasp manifold.** The pads oppose ~30 mm apart at q = 0 and the fingers cross a
  few degrees later, so graspable postures form a thin manifold: the sweep grid is
  wave-tilt (±15°) × PIP (−5…+10°), with MCP solved per point to hit tip gap = 24 mm.
  Wave tilt = opposite L/R wave values (a rigid rotation of the grasp about the pinch
  axis — it reorients gravity on the pads).
- **Boundary measurement.** Per condition (posture × mass), the minimal stable squeeze
  f* is found by a mass-scaled ascending anchor ladder followed by log-space bisection
  (7 steps, 2 seeds/probe, 3 s hold window; "stable" = drift < 2 mm in the tip frame).
  120 conditions (5 waves × 8 PIPs × 3 masses), ~2100 trials.

## 2. Core finding: the boundary is rolling-governed, not Coulomb

At sub-boundary force the object does not slide down the pads — it **rolls off the
fingertip domes** (rotation ≈ 47–170° at failure). Consequently the effective
stability ratio μ_eff = (m·g/2)/f* sits 25–35 % below the Coulomb pair value at every
tested material (§ M8b), and utilization ρ stays near 0.5 at failure: a force-cone
(Coulomb) analysis systematically over-predicts grasp capacity.

**Shape study (M8) — the direct proof.** Identical mass (50 g), identical friction
pair, identical 25 mm grasp width; only geometry varies:

| shape | contact type | f* (N) | μ_eff |
|---|---|---|---|
| disc, axis ∥ pinch axis | flat round faces | 0.318 | 0.772 |
| box 25×25×12 mm | flat square faces | 0.444 | 0.552 |
| prism 25×50×12 mm | flat faces, 2× lever arm | 0.444 | 0.552 |
| cylinder d=25, vertical axis | curved rim (line contact) | 0.483 | 0.507 |
| disc on edge (key pinch) | thin rim, gravity on the roll axis | 0.572 | 0.429 |
| sphere d=25 | point contact | 0.622 | 0.394 |

f* spans ~2× at constant Coulomb friction, ordered exactly by the object's freedom to
roll in the grasp (probe bisection resolves ±1 notch ≈ 15 %; box and prism tie within
that resolution). Lateral prehension (disc_edge) is nearly the worst case because
gravity torque acts directly along the free rolling axis.

**Material study (M8b) — friction and fingertip stiffness.** Box, 50 g, reference
posture, one attribute at a time (pair μ = harmonic mean of box and tip values):

| level | box μs/μd | pair μd | tip E (kPa) | f* (N) | μ_eff |
|---|---|---|---|---|---|
| fric_low | 0.4/0.3 | 0.45 | 50 | none — window closed | — |
| fric_base | 1.0/0.8 | 0.85 | 50 | 0.444 | 0.552 |
| fric_high | 1.5/1.2 | 1.03 | 50 | 0.318 | 0.772 |
| tip_soft | 1.0/0.8 | 0.85 | 25 | 0.526 | 0.466 |
| tip_stiff | 1.0/0.8 | 0.85 | 100 | 0.376 | 0.653 |

Three results refine the rolling story:

1. **Constant rolling discount.** In the mid-to-high friction range f* follows
   Coulomb-like ∝ 1/μ scaling, but μ_eff/μ_pair stays at ≈ 0.65–0.75: rolling always
   binds ~25–35 % below the sliding limit. Friction is not irrelevant — it sets the
   scale — but the sliding limit is never reached.
2. **Low-friction window collapse.** At pair μd = 0.45 no stable force exists at 50 g:
   the hold boundary rises (roll-off at 0.74 N) while the ejection boundary falls
   (immediate squirt-out at ≥ 1.5 N). The two boundaries cross, and the grasp becomes
   impossible at any squeeze — the friction analogue of the 120 g mass ceiling.
3. **Softer fingertips are worse, not better** (μ_eff 0.47 → 0.55 → 0.65 for
   25/50/100 kPa). The naive argument — softer pad, larger patch, more rolling
   resistance — fails: a compliant dome yields as the object rotates and provides
   less restoring moment against rolling. This matters for TacTip-style sensor
   design: sensitivity (soft skin) trades directly against grasp stability.

## 3. The equation and its envelope

Physics form fixed a priori, coefficients fitted (least squares, quadratic basis,
angles in radians):

    f1*(wave, pip, m) = (m·g/2) / μ_eff(wave, pip)
    μ_eff = 0.6508 − 0.0271·wave + 2.1668·pip − 0.0758·wave² − 7.0927·pip² + 0.3698·wave·pip

- **Fit quality (friction regime):** R² = 0.77, RMSE = 0.072 over μ_eff ∈ [0.33, 0.85]
  (n = 69 conditions). μ_eff rises smoothly with PIP (0.33 at −5° → 0.85 at +10°) and
  is nearly flat and symmetric in wave tilt.
- **Support regime (excluded from the fit, reported separately):** at PIP ≈ −2° the
  pads cradle the object and f* becomes mass-independent (0.15–0.31 N for 30→120 g;
  Coulomb predicts f* ∝ m). Weight is carried by geometry, not friction — a "sweet
  spot" posture. Regime classification is automatic: mass-scaling ratio
  f*(2m)/f*(m) < 1.5 ⇒ support.
- **Mass ceiling:** at 120 g the hold boundary from below meets the ~2.5 N
  "watermelon-seed" ejection boundary from above — the stable window is empty at
  nearly every posture. The pinch cannot robustly hold 120 g. Related: μ_eff declines
  with mass (0.75 / 0.67 at 30 / 60 g) because gravity torque worsens rolling, so mass
  extrapolation beyond the fitted range under-predicts f*.
- **Held-out validation:** at off-grid postures inside the envelope (friction regime,
  30–60 g), grasps at 1.25×f*_pred held and at 0.6×f*_pred dropped (2/2). The three
  intentional out-of-envelope probes failed exactly as the regime analysis predicts
  (support-regime posture held below prediction; 120 g dropped above it; 90 g
  extrapolation dropped above it).
- **Stability is a window, not a threshold:** squeezing ≥ ~2.5 N ejects the object
  from the dome grip. There are also two boundaries in principle: acquisition
  (bisection, includes the release transient) sits above the quasi-static minimum
  (slow force ramp-down); the map reports the acquisition boundary.

## 4. Slip detection at the TacTip's camera rate (M6)

Ground truth is kinematic (object drift > 2 mm in the tip frame; incipient slip =
tangential sliding > 5 mm/s sustained 50 ms). The detector consumes only the tactile
force signal: band energy (15 Hz one-pole high-pass, 100 ms window) of the tangential
force ‖(f2,f3)‖, threshold calibrated on a training split (3× the 99th-percentile
pre-slip floor).

| sensor model | TP | FP | FN | TN | mean latency | max latency |
|---|---|---|---|---|---|---|
| ideal (1 kHz) | 3 | 0 | 0 | 3 | 0.181 s | 0.194 s |
| TacTip camera rate (100 Hz) | 3 | 0 | 0 | 3 | 0.184 s | 0.198 s |

Downsampling to the real sensor's frame rate costs ~4 ms of latency and no accuracy:
the slip vibration signature survives the TacTip's bandwidth.

## 5. Moving grasp and the time dimension (Phase B / M7)

Carrying the 50 g box through a wave-tilt sinusoid (±10°, 0.15 Hz, 2 cycles) at
margin 1.3 over f* fails at ~10.8 s — but so does a **motionless** hold of the same
duration at the same force (~11.1 s), and the failure time is insensitive to carry
amplitude (±5/±7.5/±10°). The bisection boundary is therefore a **3-second boundary**:
just above it, the object still creeps by slow rolling and escapes on the ~10 s
scale. Rolling creep produces almost no tangential vibration until roll-off is
imminent, so ρ-monitoring gives no warning and a vibration reflex fires too late.

At margin 2.0 (0.75 N) all three controllers — fixed force, map-scheduled force, and
scheduled + vibration reflex — carry the object through the full posture sweep with
0.75 mm drift and 8° residual rotation. Along this trajectory the fitted μ_eff is
nearly flat in wave, so scheduling ≈ fixed by design; the scheduled controller's value
lies in posture excursions that change PIP or approach the regime boundaries.

**Design rule for the paper:** command f1 = 2.0 × (m·g/2)/μ_eff(θ) for sustained
holds and slow manipulation; the 1.3 margin is only sufficient for holds ≲ 5 s.

## 6. Limitations (paper section)

- The hydroelastic resultant force + centroid is an idealized TacTip; real TacTip
  output (marker flow → force estimate) adds noise and hysteresis. The 100 Hz variant
  bounds the rate effect only.
- 2 ms map trials Nyquist-limit vibration content to 250 Hz; slip-study trials use
  1 ms. URDF inertias are placeholders (rotor-inertia regularization mandatory).
- Boundary labels use a 2 mm / 3 s criterion; §5 shows the boundary location depends
  on the hold duration (creep), so f* values are tied to that window.
- Single object size (25 mm grasp width); mass validated 30–60 g. The equation's
  coefficients are fitted for the base material pair (μd_pair = 0.85, tip 50 kPa);
  M8b gives the scaling to other materials but only at the reference posture.

## 7. Artifacts

| item | file |
|---|---|
| condition dataset (120 boundaries) | `results/sweepB_conditions.csv` |
| trial dataset (~2100 trials) | `results/sweepB_trials.csv` |
| fitted equation + regimes | `results/sweepB_fit.json` |
| μ_eff / f* posture maps | `results/sweepB_mu_eff_map.png`, `sweepB_fstar_map.png` |
| example boundary | `results/sweepB_boundary_example.png` |
| held-out validation | `results/sweepB_equation_validation.json` |
| slip detector report + figures | `results/slip_eval.json`, `slip_detection_{ideal,tactip}.png` |
| Phase B ablations (margin 2.0) | `results/phaseB_summary.json`, `phaseB_{fixed,scheduled,reflex}.png` |
| Phase B failure case (margin 1.3) | `results/phaseB_summary_margin1p3.json`, `phaseB_*_margin1p3.png` |
| dynamic envelope probes | `results/phaseB_envelope.json` |
| shape study | `results/shape_{prism,cylinder,disc,disc_edge,sphere}.json` |
| material study | `results/material_{fric_low,fric_base,fric_high,tip_soft,tip_stiff}.json` |
