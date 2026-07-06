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
| disc, axis ∥ pinch axis | flat round faces | 0.298 | 0.823 |
| box 25×25×12 mm | flat square faces | 0.435 | 0.564 |
| prism 25×50×12 mm | flat faces, 2× lever arm | 0.435 | 0.564 |
| cylinder d=25, vertical axis | curved rim (line contact) | 0.464 | 0.529 |
| disc on edge (key pinch) | thin rim, gravity on the roll axis | 0.572 | 0.429 |
| sphere d=25 | point contact | 0.596 | 0.411 |

(Each probe must hold on all 5 spawn seeds; 7 bisection steps ⇒ notch ≈ 5 %.)
f* spans 2× at constant Coulomb friction, ordered exactly by the object's freedom to
roll in the grasp. Box and prism land on *exactly* the same boundary notch even at
this resolution: doubling the lever arm about the grasp axis leaves f* unchanged, so
the flat-face boundary is set by the contact patch, not by the object's inertia.
Lateral prehension (disc_edge) is nearly the worst case because gravity torque acts
directly along the free rolling axis.

**Material study (M8b) — friction and fingertip stiffness.** Box, 50 g, reference
posture, one attribute at a time (pair μ = harmonic mean of box and tip values):

| level | box μs/μd | pair μd | tip E (kPa) | f* (N) | μ_eff |
|---|---|---|---|---|---|
| fric_low | 0.4/0.3 | 0.45 | 50 | none — window closed | — |
| fric_base | 1.0/0.8 | 0.85 | 50 | 0.435 | 0.564 |
| fric_high | 1.5/1.2 | 1.03 | 50 | 0.311 | 0.789 |
| tip_soft | 1.0/0.8 | 0.85 | 25 | 0.504 | 0.487 |
| tip_stiff | 1.0/0.8 | 0.85 | 100 | 0.360 | 0.681 |

(5-seed / 7-step values, same protocol as the shape table.) Three results refine the
rolling story:

1. **Constant rolling discount.** In the mid-to-high friction range f* follows
   Coulomb-like ∝ 1/μ scaling, but μ_eff/μ_pair stays at ≈ 0.66–0.77: rolling always
   binds ~25–35 % below the sliding limit. Friction is not irrelevant — it sets the
   scale — but the sliding limit is never reached.
2. **Low-friction window collapse.** At pair μd = 0.45 no stable force exists at 50 g:
   the hold boundary rises (roll-off at 0.74 N) while the ejection boundary falls
   (immediate squirt-out at ≥ 1.5 N). The two boundaries cross, and the grasp becomes
   impossible at any squeeze — the friction analogue of the 120 g mass ceiling.
3. **Softer fingertips are worse, not better** (μ_eff 0.49 → 0.56 → 0.68 for
   25/50/100 kPa). The naive argument — softer pad, larger patch, more rolling
   resistance — fails: a compliant dome yields as the object rotates and provides
   less restoring moment against rolling. This matters for TacTip-style sensor
   design: sensitivity (soft skin) trades directly against grasp stability.

## 3. The equation and its envelope

Physics form fixed a priori, coefficients fitted by weighted least squares
(weights from the audit-measured per-mass seed noise; quadratic posture basis +
linear mass term; angles in radians, mass in kg):

    f1*(wave, pip, m) = (m·g/2) / μ_eff(wave, pip, m)
    μ_eff = 0.7069 − 0.0432·wave + 0.6019·pip + 0.0959·wave² − 1.4668·pip²
            + 0.5369·wave·pip − 4.1958·(m − 0.045)

- **Fit quality (friction regime):** R² = 0.60, RMSE = 0.054 over μ_eff ∈
  [0.56, 0.85] (n = 60 conditions; unweighted metrics on the WLS coefficients —
  OLS reaches R² = 0.75 but over-weights the noisier 60 g rows). The mass term
  (−4.2 kg⁻¹ ⇒ μ_eff drops ≈ 0.13 per 30 g) turns the previously-noted "μ_eff
  declines with mass" defect into an explicit model term; without it, and with the
  fragile pip = −5° rows excluded, a posture-only fit collapses (R² ≈ 0.36 OLS),
  i.e. within this envelope mass moves μ_eff about as much as posture does.
- **Coefficient uncertainty (bootstrap, 5000 condition resamples):** only the
  intercept (0.707, 95 % CI [0.674, 0.739]) and the mass term (−4.20, CI
  [−5.14, −3.20]) are significantly nonzero; every posture term's CI straddles
  zero (`results/stats_tests.json`). Inside the friction-regime envelope the
  boundary surface is **statistically flat in posture** — the defensible core of
  the equation is μ_eff ≈ 0.71 − 4.2·(m − 0.045), with the quadratic posture terms
  a descriptive refinement. Posture decides grasp stability at the *regime
  boundaries* (support/fragile bands, ejection), not inside the safe region — the
  earlier impression of strong PIP dependence was carried by the fragile rows.
- **Support regime (excluded, reported separately):** at PIP ≈ −2° the pads cradle
  the object and f* becomes mass-independent (0.15–0.31 N for 30→120 g; Coulomb
  predicts f* ∝ m). Weight is carried by geometry, not friction — a "sweet spot"
  posture. Classification is automatic: mass-scaling ratio f*(2m)/f*(m) < 1.5.
- **Fragile regime (excluded after the seed audit):** at PIP = −5° the boundary's
  *existence* is seed-dependent — 2/5 independent seeds cradle at f* ≈ 0.09–0.14 N
  (μ_eff 1.6–2.2, above the Coulomb pair), 3/5 find no stable anchor at all. The
  earlier posture-only fit's steep low-μ_eff end rested on these bimodal rows.
- **Mass ceiling:** at 120 g the hold boundary from below meets the ~2.5 N
  "watermelon-seed" ejection boundary from above — the stable window is empty at
  nearly every posture. The pinch cannot robustly hold 120 g, and the fitted mass
  term is validated only over 30–60 g.
- **Held-out validation: 16/16.** 16 off-grid posture/mass points inside the
  envelope (wave ±13°, PIP 0.5–8.5°, 30–60 g incl. off-grid masses), each probed at
  1.25× and 0.6× the prediction with 2 spawn seeds and all-seeds-agree scoring:
  every point held at 1.25× and dropped at 0.6× (32/32 individual probes). Of the 3
  deliberate out-of-envelope probes, the fragile-band posture (PIP = −3°) held even
  at 0.6× (cradling carries the weight — as characterized) and 120 g dropped even at
  1.25× (window closed — as characterized), while the 90 g probe **now passes**: the
  explicit mass term corrects the extrapolation failure the posture-only fit showed
  at this point, extending usable predictions at least somewhat beyond the fitted
  30–60 g range.
- **Stability is a window, not a threshold:** squeezing ≥ ~2.5 N ejects the object
  from the dome grip. There are also two boundaries in principle: acquisition
  (bisection, includes the release transient) sits above the quasi-static minimum
  (slow force ramp-down); the map reports the acquisition boundary.
- **Seed-noise audit** (12 conditions × 5 *independent* single-seed bisections,
  7 steps): in the friction regime the seed-to-seed spread of f* is 2–7 % of the
  median at 30 g (≈ the bisection notch) but 7–32 % at 60 g — at higher mass the
  boundary uncertainty is dominated by spawn-seed variability, not measurement
  resolution, consistent with the fit RMSE. Near the regime boundary the audit finds
  bimodality: at PIP = −5° only 2/5 seeds locate any stable anchor, and those land at
  f* ≈ 0.09–0.14 N (μ_eff 1.6–2.2, above the Coulomb pair — pure cradling), far below
  the 2-seed sweep's 0.34–0.62 N. Boundary *existence* is seed-dependent there. Both
  findings are folded back into the fit: the seed noise supplies the WLS weights and
  the bimodal PIP = −5° rows are excluded as the fragile regime.

## 4. Slip detection at the TacTip's camera rate (M6 / M6b)

Ground truth is kinematic (object drift > 2 mm in the tip frame; incipient slip =
tangential sliding > 5 mm/s sustained 50 ms). The detector consumes only the tactile
force signal: band energy (15 Hz one-pole high-pass, 100 ms window) of the tangential
force ‖(f2,f3)‖, threshold calibrated on a training split (3× the 99th-percentile
pre-slip floor).

The pilot study (2 postures, 6 held-out episodes) scored perfectly at both rates
(3 TP / 3 TN, latency ≈ 0.18 s) — but 6 trials cannot bound an error rate. The
expanded study runs 90 episodes (45 induced-slip, 45 clean holds) over 15
posture × mass conditions spanning the friction regime (wave ±15°, PIP 0–8°,
30–60 g), with a 30-episode training split and a 30 + 30 test split. It reverses
the calibration conclusion:

- **A single global threshold fails.** The pre-slip vibration floor varies 9×
  (ideal rate) to 72× (camera rate) across conditions, so 3× the *maximum* training
  floor — the only safe global choice — misses every test slip (0 TP / 30 FN).
  Even the oracle-best global threshold commits a false positive (30 TP / 1 FP).
- **The signature is condition-separable.** Within each condition, the slip burst
  exceeds that condition's own floor by ≥ 3.6× (median 12× ideal, 45× camera rate) in
  every episode. So the detector is sound; only the *calibration transfer* across
  conditions was broken. Two deployable schemes fix it.

**Scheduled threshold (primary).** Fit log₁₀(pre-slip floor) over (wave, pip, mass)
from the training episodes' floors — the same recipe that schedules the grip force —
and set threshold = 3× the predicted floor at the current posture and load. The floor
model is modest (R² = 0.48 ideal / 0.68 camera rate) but the ≥ 3.6× burst margin
absorbs the prediction error:

| sensor model, scheduled threshold | TP | FP | FN | TN | mean latency | miss rate (95 % UB) | FP rate (95 % UB) |
|---|---|---|---|---|---|---|---|
| ideal (1 kHz) | 30 | 1 | 0 | 29 | 0.168 s | 0 % (10 %) | 3.3 % (≈13 %) |
| TacTip camera rate (100 Hz) | 30 | 1 | 0 | 29 | 0.170 s | 0 % (10 %) | 3.3 % (≈13 %) |

This matches the oracle-best *global* threshold (which also commits 1 FP) while
needing only signals the controller already has.

**Adaptive threshold (supplement, model-free).** Alternatively, self-normalize:
threshold = max(k × running floor, floor_abs), where the running floor is a causal
trailing 2 s block-percentile with a 0.3 s guard gap (so an onsetting burst cannot
raise its own baseline), floor_abs is a sensor-noise constant (3× the quietest
training floor) that prevents ratio blow-ups on near-silent signals, detections must
be sustained 50 ms (debounce), and the detector arms only after grasp transients
settle. k is tuned on the training split (k = 100):

| sensor model, adaptive threshold | TP | FP | FN | TN | mean latency | miss rate (95 % UB) | FP rate (95 % UB) |
|---|---|---|---|---|---|---|---|
| ideal (1 kHz) | 30 | 0 | 0 | 30 | 0.166 s | 0 % (10 %) | 0 % (10 %) |
| TacTip camera rate (100 Hz) | 30 | 0 | 0 | 30 | 0.162 s | 0 % (10 %) | 0 % (10 %) |

The adaptive scheme needs no posture model at all and scored perfectly on this test
set, but its ingredients matter: without the absolute noise term it produces 19–28
false positives (ratio blow-ups on quiet holds), and without arming it fires on
grip/release transients. Per-condition calibration (thresholds from each condition's
own training episodes: 28 TP / 0 FP / 2 FN / 30 TN) remains the diagnostic upper
bound but assumes training data at every operating condition — the two schemes above
remove that assumption.

Downsampling to the real sensor's frame rate still costs essentially nothing —
in fact the camera-rate floor spread is *larger* while burst separability is
*better*, because sample-and-hold decimation suppresses the broadband floor more
than the slip burst.

## 5. Moving grasp and the time dimension (Phase B / M7)

Carrying the 50 g box through a wave-tilt sinusoid (±10°, 0.15 Hz, 2 cycles) at
margin 1.3 over f* fails — and so does a **motionless** hold at the same force. With
10 seeds per condition, margin 1.3 fails 20/20 runs and the failure time is a
distribution, not a constant: 5.4–11.1 s motionless (mean 7.8 ± 1.7 s) vs 5.3–10.8 s
moving (mean 7.6 ± 1.6 s). The two distributions are statistically indistinguishable
(Mann–Whitney U = 57, p = 0.62, n = 10 + 10) and insensitive to carry amplitude
(±5/±7.5/±10° probes) — the carry motion is not what kills the grasp. The bisection
boundary is therefore a **3-second boundary**: just above it, the object creeps by
slow rolling and escapes on the 5–11 s scale. Rolling creep produces almost no
tangential vibration until roll-off is imminent, so ρ-monitoring gives no warning
and a vibration reflex fires too late.

**Numerical control.** The creep is not integrator drift: rerunning five motionless
margin-1.3 holds at 1 ms and 0.5 ms solver steps reproduces each seed's failure time
to within 0.04 s of the other step size (e.g. 5.52 vs 5.53 s; distributions overlap
the 2 ms baseline; `results/creep_timestep_control.json`). The failure time is
converged with respect to the time step, so the creep is contact physics in the
hydroelastic model.

At margin 2.0 (0.75 N), 30/30 runs hold (10 seeds × three controllers — fixed force,
map-scheduled force, scheduled + vibration reflex) through the full posture sweep
with ≤ 0.85 mm drift. (Phase B forces were commanded from the pre-revision fit;
under the revised mass-aware equation the same 0.75 N corresponds to margin ≈ 2.1
and the failing 0.49 N to margin ≈ 1.4 — the conclusions are unchanged.) Along this trajectory the fitted μ_eff is nearly flat in wave,
so scheduling ≈ fixed by design; the scheduled controller's value lies in posture
excursions that change PIP or approach the regime boundaries.

**Design rule for the paper:** command f1 = 2.0 × (m·g/2)/μ_eff(θ) for sustained
holds and slow manipulation (0 failures in 30 seeded runs). The 1.3 margin is only
trustworthy for holds of ≲ 4 s: the earliest observed creep failure is 5.3 s.

## 6. Limitations (paper section)

- The hydroelastic resultant force + centroid is an idealized TacTip; real TacTip
  output (marker flow → force estimate) adds noise and hysteresis. The 100 Hz variant
  bounds the rate effect only.
- 2 ms map trials Nyquist-limit vibration content to 250 Hz; slip-study trials use
  1 ms. URDF inertias are placeholders (rotor-inertia regularization mandatory).
- Boundary labels use a 2 mm / 3 s criterion; §5 shows the boundary location depends
  on the hold duration (creep), so f* values are tied to that window.
- Sweep boundaries use 2 spawn seeds per probe; the audit (§3) shows the resulting
  f* is seed-limited at 60 g (spread up to ~32 %) though notch-limited at 30 g. The
  shape/material tables and Phase B rates use 5 and 10 seeds respectively.
- Slip-detector error bounds rest on 30 + 30 test episodes: a perfect score still
  only bounds each error rate below 10 % (rule of three). The scheduled threshold's
  floor model and the adaptive detector's k and floor_abs are tuned on this
  hand/object envelope; new sensors or velocity regimes need re-tuning.
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
| slip detector, pilot (6 episodes) | `results/slip_eval.json`, `slip_detection_{ideal,tactip}.png` |
| slip detector, expanded (90 episodes) | `results/slip_eval_big.json`, re-scoring `slip_eval_big_rescored.json`, cache `slip_episodes_big/` |
| Phase B ablations (margin 2.0) | `results/phaseB_summary.json`, `phaseB_{fixed,scheduled,reflex}.png` |
| Phase B failure case (margin 1.3) | `results/phaseB_summary_margin1p3.json`, `phaseB_*_margin1p3.png` |
| Phase B seed statistics (50 runs) | `results/phaseB_seeds.json` |
| dynamic envelope probes | `results/phaseB_envelope.json` |
| shape study (5 seeds / 7 steps) | `results/shape_{prism,cylinder,disc,disc_edge,sphere}.json` (2-seed originals in `results/archive_2seed/`) |
| material study (5 seeds / 7 steps) | `results/material_{fric_low,fric_base,fric_high,tip_soft,tip_stiff}.json` |
| sweep seed-noise audit | `results/sweep_audit.json` |
| statistical tests (Mann–Whitney, bootstrap CIs) | `results/stats_tests.json` |
| creep time-step control | `results/creep_timestep_control.json` |
