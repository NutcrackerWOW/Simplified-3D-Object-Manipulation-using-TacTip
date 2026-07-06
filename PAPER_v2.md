# Rolling-Governed Pinch Stability: A Validated Minimal Grip-Force Rule for Soft Tactile Fingertips

*Second draft (2026-07-06) targeting IEEE Robotics and Automation Letters (RA-L).*
*Supersedes PAPER.md (draft 1) per the grilling decisions; PAPER.md is kept unchanged as the draft-1 record.*
*Figures: `results/paper_figs/`. All numbers trace to `results/` artifacts (see the map at the end).*
*Citations are BibTeX keys from `refs.bib`; see `references.md` for the bracket map.*

---

## Abstract

Soft optical tactile fingertips such as the TacTip gain their sensitivity from
compliance, but the same compliance lets a pinched object *roll* off the fingertip
domes below the Coulomb sliding limit. In a compliant-hydroelastic simulation
(Drake) of a two-finger pinch with TacTip-style domes, we measure the minimal
stable squeeze force f\* across grasp posture, object mass, shape, friction, and
fingertip stiffness. The stability boundary is rolling-governed: the effective
stability ratio μ_eff = (mg/2)/f\* sits 25–35 % below the pair Coulomb coefficient
at every friction tested, so a friction-cone analysis systematically underestimates
the required force. Shape alone spans a 2× range of f\* at fixed friction, ordered
by the object's freedom to roll, and softer fingertips reduce μ_eff (0.68 → 0.49,
100 → 25 kPa). Bootstrap analysis shows μ_eff is statistically flat in posture
inside the validity envelope; the deployable rule is μ_eff ≈ 0.71 − 4.2(m − 0.045)
[m in kg], validated 16/16 on held-out posture–mass probes. Near the boundary,
grasps fail by slow rolling creep (5–11 s) with no vibration warning, motivating a
×2.0 force margin under which 30/30 seeded carry runs succeed; a posture-scheduled
threshold also makes a vibration slip detector deployable at the sensor's 100 Hz
camera rate. Code and data:
https://github.com/NutcrackerWOW/Simplified-3D-Object-Manipulation-using-TacTip

---

## I. Introduction

Grip force selection is the most basic decision a manipulating hand makes: squeeze
too little and the object falls, squeeze too much and the object is deformed,
ejected, or energy is wasted. Sixty years of grasp analysis provide the standard
answer — treat each contact as a friction cone, require the load wrench to be
resisted inside the cones, and add an empirical safety margin. For rigid point
contacts this is essentially correct. But modern tactile fingertips are not rigid
points: sensors such as the TacTip family [chorley2009development,
wardcherrier2018tactip] wrap a soft hemispherical membrane around a camera, and the
same compliance that gives them their sensitivity gives the grasped object a new way
to fail — it can *roll* off the dome.

This paper quantifies that failure mode and turns the result into an engineering
rule. We build a simulation testbed around a two-finger hand with compliant
hydroelastic fingertip domes, pinching gravity-loaded objects with the pinch axis
horizontal so that weight loads the contacts in shear (Fig. 1). Because the testbed
measures the *acquisition-to-hold* pipeline end to end — close, grip, release,
hold — we can locate the minimal stable squeeze force f\* by bisection for any
posture, object, and material, and ask what actually sets it.

The answer is consistent across every sweep we ran: the boundary is
**rolling-governed**. At forces just below f\* the object does not slide down the
pads; it rotates about the grasp axis and escapes (Fig. 2). The effective stability
ratio μ_eff = (mg/2)/f\* — the number a Coulomb analysis would call "friction being
used"; higher μ_eff means a cheaper-to-hold grasp — saturates 25–35 % below the
material pair's Coulomb coefficient at every tested friction level, and a shape
study at constant mass and friction spans a 2× range of f\* ordered exactly by
rolling freedom (Fig. 3). None of this is visible to a friction-cone analysis, which
predicts the same force for a sphere and a disc.

Rolling failure of soft contacts is not new *qualitatively* — the soft-finger
contact model and its limit surface [goyal1991planar, kao1992quasistatic,
xydas1999modeling] bound the coupled force–torque capacity of a compliant patch, and
the grasping literature has long known that torsional load eats tangential capacity.
Our contribution is *quantitative and operational*: we measure where the boundary
actually lies for a realistic soft-fingertip geometry across posture, mass, shape,
friction, and stiffness; we identify the regimes where the friction picture breaks
entirely (a geometric support regime, a fragile seed-dependent band, a squeeze-out
ejection ceiling, and a mass ceiling where the stable window closes); and we
compress the result into a one-line minimal-force rule with a stated envelope,
tested on held-out conditions.

Contributions:

1. **Mechanism, quantified.** Across shape, friction, and stiffness sweeps, the
   pinch stability boundary for soft hemispherical fingertips is set by rolling, not
   Coulomb sliding: μ_eff sits 25–35 % below the pair friction at all tested
   frictions; geometry alone moves f\* by 2× at fixed friction; and softer
   fingertips lower μ_eff, from 0.68 at 100 kPa to 0.49 at 25 kPa, exposing a direct
   sensitivity-versus-stability trade-off in soft tactile sensor design.
2. **A validated minimal grip-force rule.** f₁\* = (mg/2)/μ_eff with the one-line
   μ_eff ≈ 0.71 − 4.2(m − 0.045): bootstrap confidence intervals show the boundary
   is statistically flat in posture inside the validity envelope, so mass — not
   posture — carries the rule, with a fitted quadratic posture refinement and
   automatically classified failure regimes reported alongside. The rule scores
   16/16 on held-out posture–mass probes (hold at 1.25× prediction, drop at 0.6×).
3. **Deployable slip-threshold calibration at the sensor's camera rate.** On a
   90-episode study spanning the envelope, the pre-slip vibration floor varies up to
   72× across conditions, so any single global threshold fails (0/30 detections at
   the only safe setting) even though every slip burst clears its own condition's
   floor by ≥ 3.6×. Scheduling the threshold over (φ, θ, m) — the same recipe that
   schedules grip force — recovers 30 TP / 1 FP / 0 FN / 29 TN at both the ideal
   rate and the 100 Hz camera rate.

We also show the measured boundary is **hold-duration-dependent**: at 1.3× f\* the
grasp survives the classic 3 s test window but fails between 5.3 and 11.1 s by slow
rolling creep, with no detectable difference between motionless and carried holds
(§VII), and with failure times converged in the solver time step. Rolling creep
produces almost no tangential vibration until roll-off is imminent, so no reflex can
save a grasp commanded at that margin — the ×2.0 margin in our rule is not
conservatism but a measured requirement for sustained holds.

The study is simulation-only by design: hydroelastic contact gives us a controlled,
converged testbed in which f\* is measurable to a few percent, every failure is
replayable, and 2,100-trial sweeps are tractable. Hardware transfer — replacing the
idealized contact resultant with real TacTip marker flow — is future work, and §VIII
details what the idealization does and does not cover.

## II. Related Work

**Soft-finger contact and limit surfaces.** The soft-finger contact model augments
point friction with torsional resistance over a finite patch; the limit-surface
formalism [goyal1991planar, kao1992quasistatic, xydas1999modeling] bounds the
coupled tangential-force/torque capacity and predicts that torque demand reduces
tangential capacity. Our measurements are consistent with that picture and go beyond
it in two ways: the failing rotation here is about the *grasp axis* (the object
rolls over the dome's curvature, a geometric escape rather than in-patch spin), and
we measure the resulting boundary as a deployable scalar field μ_eff(φ, θ, m) rather
than a per-contact constitutive model.

**Grip force control and margins.** Human studies [johansson1984roles] established
grip-force scheduling with a small safety margin above the slip limit, and robotic
equivalents estimate the friction coefficient or slip margin online from tactile
signals, from early skin-acceleration slip sensing [howe1989sensing] to recent
soft-finger Coulomb-state estimation from tactile arrays [jang2024soft]. These
approaches presume the operative limit is frictional and *estimate* where it lies;
we instead measure that for soft domes the operative limit is rolling, sits 25–35 %
below the frictional one, and — near the boundary — moves with hold duration, and we
*prescribe* the resulting minimal force. Margins calibrated against Coulomb slip are
optimistic in exactly the regime where they matter.

**Slip detection with optical tactile sensors.** Vibration- and flow-based slip
detection is well established for TacTip [chorley2009development,
wardcherrier2018tactip, james2018slip] and GelSight-class sensors
[johnson2009retrographic, yuan2017gelsight]; for a survey of tactile sensing in
dexterous hands see [kappassov2015tactile]. Reported detectors are typically
evaluated at one or few conditions. Our expanded study reproduces the common
single-condition success and then shows the cross-condition calibration problem
(72× floor variation) that a deployment must solve; the proposed fix (threshold
scheduling over posture and load) uses no additional sensing.

**Simulation of compliant contact.** We rely on Drake's compliant-hydroelastic
contact model with the SAP discrete solver [elandt2019pressure,
masterjohn2022velocity, castro2023unconstrained], which produces continuous
pressure-field contact wrenches for soft geometries and is a current standard for
contact-rich manipulation simulation. Tactile simulators in the same spirit have
demonstrated sim-to-real transfer for TacTip-class sensors [lin2022tactilegym2],
which is the intended path for the hardware validation discussed in §VIII. We verify
our headline temporal result (creep failure time) is converged across 2×–4×
time-step refinement.

## III. Simulation Testbed

**Hand and materials.** The testbed (Fig. 1a) is a two-finger hand ("Touch_Finger")
with per-finger kinematic chain wave → MCP → PIP → DIP, mounted fingers-down so the
pinch axis is horizontal and gravity loads the contacts in shear. Wave, MCP, and PIP
are actuated (posture PD with gravity feedforward); DIP is passive, a torsional
spring toward the measured PIP angle (k = 0.35 N·m/rad, ≈ 5° of give at 1–2 N),
mimicking the compliance of a distal pad mount. Fingertips are hemispherical domes
simulated as compliant hydroelastic solids (elastic modulus 50 kPa nominal,
Hunt–Crossley dissipation 1.5 s/m, μs/μd = 1.2/0.9); objects are rigid hydroelastic
(μs/μd = 1.0/0.8), giving a Coulomb pair coefficient μd ≈ 0.85 (harmonic mean). The
reference object is a 25 × 25 × 12 mm, 50 g block. Simulation uses Drake 1.54, SAP
solver, 2 ms steps for boundary trials and 1 ms for slip trials.

**Idealized tactile signal.** Per fingertip we read the hydroelastic contact
resultant force and patch centroid — an idealized TacTip. From the two centroids we
build the grasp frame (Fig. 1b): e₁ along the grasp axis (squeeze force f₁),
e₂ vertical in the plane normal to e₁ (the weight-bearing shear f₂), e₃ = e₁ × e₂
(lateral shear f₃). Utilization ρ = ‖(f₂, f₃)‖/f₁ is the fraction of a Coulomb
budget in use; classical analysis predicts failure at ρ ≈ μ_pair. Sensor-rate
realism is covered by evaluating the slip detector on a 100 Hz sample-and-hold
variant of the signal, the camera rate of the physical sensor.

**Grasp acquisition and force control.** A state machine closes to a posture,
acquires contact, servos the squeeze component f₁ to a setpoint (integral torque
trim on MCP), releases the object's spawn support, and holds. Gravity-compensation
feedforward on the actuated joints is required at tilted postures — without it, PD
sag consumes the few-degree contact window.

**The grasp manifold is thin.** The pads oppose ~30 mm apart at zero flexion and the
fingers physically cross a few degrees later (Fig. 1c): the pad gap sweeps its full
graspable band in ≈ 3° of MCP flexion. Grasp postures are therefore parameterized by
**grasp tilt φ** (±15°; opposite wave values on the two fingers, a rigid rotation of
the grasp about the pinch axis that reorients gravity on the pads) and **PIP flexion
θ** (−5° … +10°), with MCP solved per condition to hit a 24 mm pad gap.

**Boundary protocol.** For each condition (φ, θ, mass) the minimal stable squeeze
f\* is located by a mass-scaled ascending force ladder followed by log-space
bisection: 7 steps, 2 spawn seeds per probe (a probe is stable only if *all* seeds
hold), "stable" = object drift < 2 mm in the tip frame over a 3 s hold. The primary
sweep covers 5 tilts × 8 flexions × 3 masses (30/60/120 g) = 120 conditions, ≈ 2,100
trials. Three dedicated studies use stronger statistics: shape/material tables use 5
seeds per probe, the carried-grasp failure statistics use 10 seeds per condition,
and a seed-noise audit (12 conditions × 5 independent single-seed bisections)
measures the seed-to-seed spread of f\* itself — 2–7 % of the median at 30 g (the
bisection notch) but 7–32 % at 60 g, numbers that become the weights and exclusions
of §V.

## IV. The Stability Boundary Is Rolling-Governed

**Failure looks like rolling, everywhere.** Fig. 2 shows the canonical failure: a
50 g block held at 1.3× f\* at the reference posture. For five seconds the grasp
looks quiet — drift under a millimeter — while the block rotates at fractions of a
degree per second about the lateral axis. The rotation accelerates smoothly, the
block rolls over the dome shoulders, and it is gone ~200 ms later, tumbling 176°
before free fall. Translation (the thing a Coulomb analysis watches) stays below the
2 mm threshold until the roll-off is already unrecoverable. At failure, utilization
ρ ≈ 0.5: the grasp "uses" barely half its nominal friction budget when it dies.

**The stability ratio never reaches Coulomb.** Define μ_eff = (mg/2)/f\*, the value
a friction-cone analysis would infer from the measured boundary; higher μ_eff means
less force is needed — a better grasp. If sliding governed, μ_eff would equal the
pair coefficient. Instead, across the friction sweep (box object, reference posture,
Table below and Fig. 3b), μ_eff/μ_pair stays at 0.66–0.77 in the mid-to-high
friction range: f\* follows the Coulomb-like 1/μ scaling — friction sets the
*scale* — but the boundary binds 25–35 % early, at the roll-off.

| material variation | pair μd | f\* (N) | μ_eff | μ_eff/μ_pair |
|---|---|---|---|---|
| low friction (box μd 0.3) | 0.45 | none — window closed | — | — |
| base (box μd 0.8) | 0.85 | 0.435 | 0.564 | 0.66 |
| high friction (box μd 1.2) | 1.03 | 0.311 | 0.789 | 0.77 |
| soft tips, 25 kPa | 0.85 | 0.504 | 0.487 | 0.57 |
| stiff tips, 100 kPa | 0.85 | 0.360 | 0.681 | 0.80 |

**Geometry alone spans 2× at fixed friction.** The shape study (Fig. 3a) holds mass
(50 g), friction pair, and grasp width (25 mm) fixed and varies only geometry:

| shape | contact type | f\* (N) | μ_eff |
|---|---|---|---|
| disc, axis ∥ grasp axis | flat round faces | 0.298 | 0.823 |
| box 25 × 25 × 12 mm | flat square faces | 0.435 | 0.564 |
| prism 25 × 50 × 12 mm | flat faces, 2× lever arm | 0.435 | 0.564 |
| cylinder d = 25 mm, axis vertical | curved rim | 0.464 | 0.529 |
| disc on edge (key pinch) | thin rim, gravity on roll axis | 0.572 | 0.429 |
| sphere d = 25 mm | point contact | 0.596 | 0.411 |

The ordering is exactly the object's freedom to roll in the grasp: flat faces
pressed by domes resist rotation best; rims and points roll easily; lateral
prehension of a disc is nearly worst because gravity torque acts directly about the
free rolling axis. Two details sharpen the mechanism. First, box and prism land on
*exactly* the same bisection notch (f\* = 0.435 N at 5-seed resolution): doubling
the lever arm about the grasp axis changes nothing, so the flat-face boundary is set
by the contact patch, not the object's inertia. Second, the span disc → sphere is
2.0× in f\* — larger than the entire friction effect over the tested range — at
*identical* Coulomb friction. No friction-cone analysis, however carefully
calibrated, can reproduce this column.

**Softer fingertips are worse.** The naive expectation — softer pad, larger patch,
more rolling resistance — fails in the direction that matters for sensor design:
μ_eff falls from 0.68 (100 kPa) to 0.56 (50 kPa) to 0.49 (25 kPa). A compliant dome
yields as the object rotates and provides less restoring moment against rolling, so
the skin softness that maximizes tactile sensitivity directly taxes grasp stability.
For TacTip-class sensors this is a quantified design trade-off, not a qualitative
caution.

**The friction picture can collapse entirely.** At pair μd = 0.45 the hold boundary
(roll-off from below, 0.74 N) rises past the ejection boundary (squeeze-out from
above, ≥ 1.5 N at this friction): the stable window is empty and *no* squeeze force
holds a 50 g box. Stability for soft fingertips is a **window**, not a threshold —
its ceiling (§V) is as real as its floor.

## V. The Stability Envelope and a Minimal Grip-Force Rule

**The rule.** Inside the validity envelope established below, the minimal stable
squeeze force is

    f₁*(m) = (m·g/2) / μ_eff,      μ_eff ≈ 0.71 − 4.2·(m − 0.045)      [m in kg]

Mass, not posture, carries the rule. Bootstrap confidence intervals over the fitted
sweep (5,000 condition resamples) retain only the intercept (0.707, 95 % CI
[0.674, 0.739]) and the mass slope (−4.20, CI [−5.14, −3.20]); every posture
coefficient's CI straddles zero. Inside the envelope, μ_eff is statistically flat in
posture; posture decides grasp stability at the *regime boundaries* (below), not
inside the safe region.

**Descriptive posture refinement.** The full weighted-least-squares fit over the
**friction regime** (n = 60 conditions; regimes below), in the physics form fixed a
priori,

    μ_eff(φ, θ, m) = 0.7069 − 0.0432·φ + 0.6019·θ + 0.0959·φ² − 1.4668·θ²
                     + 0.5369·φ·θ − 4.1958·(m − 0.045)      [φ, θ in rad; m in kg]

uses per-mass weights from the audit's measured seed noise (σ_rel = 1.5 % at 30 g,
8.9 % at 60 g) and reaches R² = 0.60, RMSE = 0.054 over μ_eff ∈ [0.56, 0.85]
unweighted (OLS reaches R² = 0.75 but over-weights the noisier 60 g rows). The
explicit mass term — μ_eff drops ≈ 0.13 per 30 g — is essential: with the fragile
rows excluded, a posture-only fit collapses to R² ≈ 0.36, i.e. inside the envelope
mass moves the boundary about as much as posture does. The strong PIP-flexion
dependence suggested by the raw map (Fig. 4a,b) is carried by rows the regime
analysis removes. Over the 16 held-out validation conditions the one-line rule and
the full quadratic differ by at most +9 % / −0.3 % in predicted f\*
(`results/one_liner_vs_quadratic.json`) — far inside the 0.6×–1.25× experimental
probe brackets — so the validation below applies to the one-line rule.

**Regimes (Fig. 4a,b).** Two bands are excluded from the fit and reported
separately, both detected automatically:

- *Support regime* (θ ≈ −2°, hatched ///): the slightly-everted pads cradle the
  object and f\* becomes mass-independent (0.15–0.31 N for 30 → 120 g, where Coulomb
  predicts proportionality) — weight is carried by geometry, not friction.
  Classifier: mass-scaling ratio f\*(2m)/f\*(m) < 1.5.
- *Fragile regime* (θ = −5°, hatched ×××): the boundary's *existence* is
  seed-dependent. In the audit, 2/5 independent seeds cradle at f\* ≈ 0.09–0.14 N
  (μ_eff 1.6–2.2 — above the Coulomb pair, pure cradling) and 3/5 find no stable
  force at all. These bimodal rows are excluded; they had carried the steep
  low-μ_eff end of earlier posture-only fits.

**The window and its ceiling (Fig. 4c).** Squeezing beyond ≈ 2.5 N ejects the object
from between the domes (squeeze-out ejection) regardless of posture. The stable
window [f\*, 2.5 N] narrows with mass — the boundary curve rises as μ_eff falls —
and at 120 g it is empty at nearly every posture: the pinch cannot robustly hold
120 g, and the fitted mass term is validated only over 30–60 g.

**Held-out validation: 16/16.** Sixteen off-grid posture/mass points inside the
envelope (tilt up to ±13°, flexion 0.5–8.5°, masses 30–60 g including off-grid
values) were each probed at 1.25× and 0.6× the predicted f\*, two spawn seeds per
probe, all-seeds-agree scoring. All 16 held at 1.25× and dropped at 0.6× (32/32
probes). Three deliberate out-of-envelope probes behaved as characterized: a
fragile-band posture held even at 0.6× (cradling), 120 g dropped even at 1.25×
(window closed), and a 90 g probe passed — the mass term extends usable prediction
at least somewhat beyond the fitted range.

## VI. Slip Detection at the Sensor's Camera Rate

A grip-force rule wants a safety net: detect incipient slip and bump the setpoint.
The detector is deliberately simple — band energy (15 Hz one-pole high-pass, 100 ms
window) of the tangential force ‖(f₂, f₃)‖, thresholded — and is evaluated against
kinematic ground truth (drift > 2 mm; incipient slip = tangential sliding > 5 mm/s
sustained 50 ms) at both the ideal simulation rate and a 100 Hz sample-and-hold
camera-rate variant matching the physical sensor.

A pilot at a single posture scores perfectly at both rates, reproducing the common
literature result. The deployable question appears at scale. The expanded study runs
90 episodes (45 induced-slip by force ramp-down, 45 clean holds) over 15
posture × mass conditions spanning the envelope, split 30 train / 60 test:

- **A single global threshold fails.** The pre-slip vibration floor varies 9×
  (ideal) to 72× (camera rate) across conditions. The only safe global choice — 3×
  the maximum training floor — misses every test slip (0 TP / 30 FN); even the
  oracle-best global threshold commits a false positive.
- **Detection itself is sound.** Within each condition, every slip burst clears that
  condition's own floor by ≥ 3.6× (median 12× ideal, 45× camera rate). The broken
  piece is *calibration transfer* across conditions (Fig. 5a,b).

**Scheduled threshold.** Fit log₁₀(pre-slip floor) over (φ, θ, m) on the training
episodes — the same scheduling recipe as the grip-force rule — and set the threshold
to 3× the predicted floor at the current posture and load. The floor model is modest
(R² = 0.48 ideal, 0.68 camera rate) but the ≥ 3.6× burst margin absorbs the
prediction error:

| scheduled threshold | TP | FP | FN | TN | mean latency | miss rate (95 % UB) | FP rate (95 % UB) |
|---|---|---|---|---|---|---|---|
| ideal (1 kHz) | 30 | 1 | 0 | 29 | 0.168 s | 0 % (10 %) | 3.3 % (≈13 %) |
| camera rate (100 Hz) | 30 | 1 | 0 | 29 | 0.170 s | 0 % (10 %) | 3.3 % (≈13 %) |

This matches the oracle-best global threshold while requiring only signals the
controller already has (Fig. 5c). A model-free adaptive variant — a running-floor
self-normalization guarded by an absolute sensor-noise term and post-grasp arming —
reaches a perfect 30/0/0/30 on this test set at both rates; its construction,
ablations (each guard is necessary), and tuning are documented in the released code.
Per-condition calibration (28/0/2/30) remains a diagnostic upper bound but assumes
training data at every operating condition — precisely the assumption scheduling
removes.

Camera-rate operation costs essentially nothing: the floor *spread* worsens but
burst separability *improves*, because sample-and-hold decimation suppresses the
broadband floor more than the slip burst.

## VII. Carried Grasps and the Time Dimension

**The boundary is hold-duration-dependent.** Carrying the 50 g box through a grasp
tilt sinusoid (±10°, 0.15 Hz, 2 cycles) at 1.3× f\* fails — and so does a motionless
hold at the same force. With 10 seeds per condition, margin 1.3 fails 20/20 runs,
and the failure time is a distribution, not a constant: 5.4–11.1 s motionless
(7.8 ± 1.7 s) versus 5.3–10.8 s moving (7.6 ± 1.6 s). We detect no difference
between the motionless and carried distributions (Mann–Whitney U = 57, p = 0.62;
with n = 10 per group only large differences would be detectable; Fig. 6a), and the
failure times are insensitive to carry amplitude — the motion is not what kills the
grasp. The mechanism is the same slow rolling creep of Fig. 2: a grasp that passes
any fixed-duration stability test just above f\* still escapes on the 5–11 s scale.
Rolling creep produces almost no tangential vibration until roll-off is imminent, so
utilization monitoring gives no warning and a vibration reflex fires too late — no
detector can rescue a grasp commanded at margin 1.3.

**The creep is physics, not solver drift.** Rerunning five motionless margin-1.3
holds at 1 ms and 0.5 ms time steps reproduces each seed's failure time to within
0.04 s of the other step size (Fig. 6b); the failure time is converged with respect
to the time step in the hydroelastic model.

**Design rule.** Command **f₁ = 2.0 × (mg/2)/μ_eff** for sustained holds and slow
manipulation. At this margin, 30/30 seeded carry runs (10 seeds × three
controllers — fixed force, map-scheduled, scheduled + vibration reflex) complete the
full posture sweep with ≤ 0.85 mm drift. The 1.3 margin is trustworthy only for
holds ≲ 4 s (earliest observed creep failure: 5.3 s). Along the tested trajectory
the fitted μ_eff is nearly flat in tilt — consistent with §V's flatness result — so
scheduling ≈ fixed force by design there; the scheduled controller earns its keep in
excursions that change PIP flexion or approach regime boundaries. (Phase-B forces
were commanded from a pre-revision fit; under the final equation the same 0.75 N is
margin ≈ 2.1 and the failing 0.49 N is margin ≈ 1.4 — conclusions unchanged.)

## VIII. Limitations and Conclusion

**Limitations.** (i) The tactile signal is an idealized TacTip — the hydroelastic
contact resultant and centroid; real marker-flow force estimates add noise,
hysteresis, and calibration error. The 100 Hz variant bounds only the rate effect.
Hardware validation, following the sim-to-real path demonstrated for TacTip-class
sensors [lin2022tactilegym2], is the natural next step. (ii) Boundary trials at 2 ms
Nyquist-limit vibration content to 250 Hz; slip trials use 1 ms. (iii) The 2 mm /
3 s stability criterion itself defines f\*, and §VII shows the boundary moves with
hold duration; f\* values are tied to that window (the ×2.0 rule is the
duration-robust statement). (iv) Sweep boundaries use 2 seeds per probe and are
seed-limited at 60 g (spread to 32 %); the shape/material tables and carry
statistics use 5 and 10 seeds. (v) A perfect 30 + 30 slip score still only bounds
each error rate below ~10 % (rule of three); the floor model and adaptive constants
are tuned to this hand/object envelope. (vi) One grasp width (25 mm) and one base
material pair; masses validated 30–60 g; the material study gives scaling to other
pairs only at the reference posture.

**Conclusion.** For soft tactile fingertips, the minimal-grip-force question is not
a friction question. The stability boundary is set by rolling: it binds 25–35 %
below the Coulomb limit at every friction tested, moves 2× with shape at fixed
friction, worsens as fingertips soften, and — near the boundary — depends on how
long you intend to hold. All of this is invisible to a friction-cone analysis and
all of it is capturable: a one-line mass-dependent rule with a measured envelope
predicts held-out boundaries 16/16, a posture-scheduled threshold makes a simple
vibration detector deployable at the sensor's real frame rate, and a ×2.0 margin
converts the rule into a controller that did not drop an object in 30 seeded
carries. This paper contributes that measured map — the flat-in-posture μ_eff, the
support and fragile regimes, the sensitivity–stability trade-off — and the validated
rule distilled from it, with every number reproducible from the released code and
data.

---

## Figure map

| paper figure | file (results/paper_figs/) | content |
|---|---|---|
| Fig. 1 | `F1_setup.png` | testbed, grasp frame/utilization, thin grasp manifold |
| Fig. 2 | `F2_rolling.png` | rolling-creep failure anatomy (snapshots + rotation/drift traces) |
| Fig. 3 | `F3_shapes_materials.png` | shape and material μ_eff bars vs Coulomb pair |
| Fig. 4 | `F4_envelope.png` | μ_eff maps (30/60 g) with regimes; force window vs mass |
| Fig. 5 | `F6_slip.png` | slip episode + 72× floor spread + errors by calibration scheme |
| Fig. 6 | `F5_creep.png` | creep failure-time distributions (static vs moving); time-step control |

*(File names reflect build order; paper numbering follows section order.)*

## Changes from draft 1 (grilling decisions Q2–Q7)

- Abstract rewritten to ≈ 190 words, quantitative framing ("systematically
  underestimates the required force" replaces "is wrong"), slip demoted to one
  clause, repo URL filled in.
- One-line rule promoted to the headline (§V leads with it; Contribution 2
  restructured); quadratic fit demoted to descriptive refinement with the new
  one-liner-vs-quadratic agreement check (+9 % / −0.3 % over the 16 validation
  conditions, `scripts/check_one_liner.py`).
- §VI: scheduled threshold is the sole presented method; adaptive variant compressed
  to three sentences pointing to the repo.
- §VII / §I: Mann–Whitney claim reframed descriptively ("we detect no difference…
  only large differences would be detectable"), replacing "statistically
  indistinguishable".
- Conclusion: "first deployable characterization" priority claim dropped; claims the
  measured map + validated rule.
- μ_eff orientation ("higher = cheaper to hold / better") stated at both
  definitions; stiffness arrows now always quoted in the falling direction
  (0.68 → 0.49 as tips soften).
- Bracketed placeholder citations replaced with verified `refs.bib` keys, including
  the new estimate-vs-prescribe contrast [jang2024soft] and sim-to-real path
  [lin2022tactilegym2].

## Pre-submission checklist

- [x] Reference list built and verified (`refs.bib`, 16 entries; `references.md`).
- [x] Repository URL filled in (public: NutcrackerWOW/Simplified-3D-Object-Manipulation-using-TacTip).
- [x] Fingertip design is the author's own — no third-party URDF/mesh rights needed.
- [ ] Port to the RA-L LaTeX template in `paper/`; recheck figure sizing (3.5 in / 7.16 in columns).
- [ ] Re-verify flagged citation fields at compile time (Howe 1989 pages; Masterjohn DOI; Chorley pages).
- [ ] Author list, affiliations, acknowledgments (needs real name + affiliation).
- [ ] LICENSE (BSD-3-Clause code, CC-BY-4.0 data/model) + repro instructions in the repo.
