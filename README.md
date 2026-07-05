# pinchlab — 3D Object Manipulation, Pinching and Slipping Using Tactile Sensors

Drake simulation study on the Touch_Finger 2-finger hand with TacTip-style compliant
hydroelastic fingertips. Pipeline: force-controlled pinch trials → minimal-stable-force
sweep across grasp postures → fitted stability boundary → closed-form grip-force equation →
slip detection from force vibration → moving-grasp validation.

## Model

`model/Touch_Finger/Complete.urdf` — now ships complete for Drake:
revolute joints with limits (MCP/PIP/DIP −5°…+90°, Wave ±15°), both tips compliant
hydroelastic (E=50 kPa, Hunt-Crossley 1.5, μs/μd = 1.2/0.9), `.obj` mesh references.
`pinchlab/model.py` re-applies these settings idempotently at load, so
`pinchlab.params.ModelParams` remains the runtime source of truth for material sweeps.

Joint semantics (per the project design):
- **Wave, MCP, PIP actively actuated** (posture PD + gravity feedforward).
- **DIP is passive**: torsional spring toward the measured PIP angle
  (k = 0.35 N·m/rad → ~5° give at 1–2 N tip force).
- **Squeeze force** is regulated by an outer tactile loop: the hydroelastic contact
  force is decomposed and the squeeze component f1 is servoed to a setpoint by an
  integral torque trim on MCP.

## Geometry facts (measured — they shape the experiment design)

- The pads oppose ~30 mm apart at q = 0; a 25 mm object is contacted after only a few
  degrees of flex, and the fingers *cross* beyond that. Grasp postures form a thin
  manifold: the sweep grid is (wave-tilt × PIP) with **MCP solved** per point
  (`HandKinematics.solve_mcp_for_gap`).
- "Wave tilt" = opposite L/R wave values (both fingers tilt together about the pinch
  axis; full ±15° usable). Equal values splay the tips apart (infeasible beyond ~±4°).
- The object is a 25×25×12 mm block (a full 25 mm cube jams its top corners into the
  Distal links above the sensing pads), spawned 3 mm below the dome witness midpoint.

## Force decomposition (the tactile signal)

Per finger: contact patch centroid p and resultant force F from hydroelastic contact.
Grasp axis d = (p_R − p_L)/‖·‖; basis e1 = ±d (squeeze), e2 = vertical ⊥ d
(weight-carrying shear), e3 = d × e2 (lateral). Utilization ρ = ‖(f2,f3)‖/f1.
Stability = ρ below an *effective* bound μ_eff — measured, not assumed: the dominant
failure at low force is the box **rolling** off the domes, so μ_eff ≈ 0.5, well below
the Coulomb pair value 1.09. Stability is also a **window** in force: squeezing ≥2.5 N
ejects the box from the dome grip.

## The equation (paper product)

    f1*(wave, pip, m) = (m·g/2) / μ_eff(wave, pip)
    μ_eff = c0 + c1·wave + c2·pip + c3·wave² + c4·pip² + c5·wave·pip

Coefficients, fit diagnostics, and validation live in `results/<tag>_fit.json` after
running the pipeline.

### Measured results (sweepB, 120 conditions)

The posture map splits into **two regimes**, classified automatically by mass scaling
(`classify_regimes`: Coulomb predicts f*(2m)/f*(m)=2; ratio <1.5 ⇒ support):

- **Friction regime** (all postures except pip ≈ −2°): the quadratic μ_eff fit reaches
  R² = 0.77, RMSE = 0.072 over μ_eff 0.33–0.85. μ_eff rises smoothly with PIP
  (0.33 at −5° → 0.85 at +10°), is symmetric in wave tilt, and *declines with mass*
  (0.75/0.67 at 30/60 g) — rolling failure worsens under gravity torque.
- **Support regime** (pip ≈ −2°): the pads cradle the box; f* is mass-independent
  (0.15–0.31 N), so weight is carried by geometry, not friction — μ_eff stops measuring
  friction there and those rows are excluded from the fit (kept in the JSON as
  `support_regime`).

Additional envelope limits: **120 g exceeds the pinch capacity** at nearly all postures
(the hold boundary from below meets the ~2.5 N ejection boundary from above → empty
window), and mass extrapolation beyond 60 g under-predicts f* (held-out 90 g point
failed at 1.25×f*). Held-out validation: 2/2 inside the fitted envelope; all misses
map onto these characterized limits.

Slip detector (M6): on the held-out split, 3 TP / 0 FP / 0 FN / 3 TN at **both** the
ideal rate and the 100 Hz TacTip camera rate; detection latency ≈ 0.18 s after
kinematic slip onset (downsampling to camera rate costs only ~4 ms).

### Phase B: the boundary is a 3-second boundary (rolling creep)

Carrying the 50 g box through wave ±10° @ 0.15 Hz at margin 1.3 over the fitted f*
drops the box at ~10.8 s — but so does a **motionless** 14.3 s hold at the same force
(drop at ~11.1 s), and amplitude ±5°/±7.5°/±10° barely shifts the failure time. The
bisection boundary (3 s hold window) is therefore **duration-dependent**: sub-margin
squeeze lets the box roll off by slow creep, with utilization ρ ≈ 0.5 giving no
warning and the vibration reflex firing too late for abrupt roll-off. At **margin 2.0**
(0.75 N) all three controllers (fixed / scheduled / scheduled+reflex) carry the box
through the full wave sweep with 0.75 mm drift and 8° rotation
(`results/phaseB_summary.json`, envelope probes in `results/phaseB_envelope.json`,
margin-1.3 failures archived as `phaseB_*_margin1p3.*`).

### Shape study (M8): μ_eff tracks rolling freedom, not friction

Same mass (50 g), same friction pair (μ 1.0/0.8), same 25 mm grasp width, reference
posture — only geometry differs (`scripts/eval_shapes.py`, `results/shape_*.json`):

| shape                            | contact                | f* (N) | μ_eff |
|----------------------------------|------------------------|--------|-------|
| disc (axis ∥ pinch axis)         | flat round faces       | 0.318  | 0.772 |
| box 25×25×12                     | flat square faces      | 0.444  | 0.552 |
| prism 25×50×12 (2× lever arm)    | flat square faces      | 0.444  | 0.552 |
| cylinder (vertical axis)         | curved rim (line)      | 0.483  | 0.507 |
| disc_edge (key pinch, upright)   | thin rim, gravity-roll | 0.572  | 0.429 |
| sphere d=25                      | point                  | 0.622  | 0.394 |

Coulomb friction is identical across rows, yet f* spans ~2× and orders exactly by the
object's freedom to roll in the grasp — direct evidence that the stability boundary is
**rolling-governed**, the paper's central claim. The `disc_edge` row is the lateral-
prehension (key-pinch) configuration: the disc stands like a wheel and gravity torque
acts directly along the free rolling axis. (Probe bisection resolves ~±1 notch ≈ 15 %;
box and prism tie within that resolution. The box value here is the 50 g probe —
the sweep-fitted μ_eff at this posture is 0.65 because μ_eff declines with mass.)

### Material study (M8b): friction scaling, window collapse, and a stiffness surprise

Box, 50 g, reference posture, one attribute varied at a time
(`scripts/eval_materials.py`, `results/material_*.json`; pair μ = harmonic mean of
box and tip values):

| level      | box μs/μd | pair μd | tip E (kPa) | f* (N) | μ_eff |
|------------|-----------|---------|-------------|--------|-------|
| fric_low   | 0.4/0.3   | 0.45    | 50          | — none | —     |
| fric_base  | 1.0/0.8   | 0.85    | 50          | 0.444  | 0.552 |
| fric_high  | 1.5/1.2   | 1.03    | 50          | 0.318  | 0.772 |
| tip_soft   | 1.0/0.8   | 0.85    | 25          | 0.526  | 0.466 |
| tip_stiff  | 1.0/0.8   | 0.85    | 100         | 0.376  | 0.653 |

Three findings: (1) in the mid-to-high friction range f* scales roughly like Coulomb
(∝ 1/μ) but with a **constant rolling discount** — μ_eff/μ_pair ≈ 0.65–0.75, i.e.
rolling binds ~25–35 % below the sliding limit; (2) at low friction the stability
window **collapses entirely** — the hold boundary rises (roll-off at 0.74 N) while
the ejection boundary falls (squirt-out at ≥ 1.5 N), leaving no stable force at 50 g;
(3) **softer fingertips are worse, not better** (μ_eff 0.47 → 0.55 → 0.65 across
25/50/100 kPa): the naive bigger-patch argument fails — a compliant dome yields as the
object rotates, offering less restoring moment against rolling.

## Running the pipeline

```bash
.venv/bin/python scripts/run_trial.py --force 0.6 --meshcat   # one visual trial
.venv/bin/python scripts/run_sweep.py --tag sweepB            # Phase A sweep (parallel, resumable)
.venv/bin/python scripts/fit_map.py --tag sweepB              # μ_eff map + equation + figures
.venv/bin/python scripts/eval_slip.py                         # slip detector: ideal + TacTip 100 Hz
.venv/bin/python scripts/validate_equation.py --tag sweepB    # held-out prediction test
.venv/bin/python scripts/validate_moving.py --tag sweepB --margin 2.0  # Phase B carry
.venv/bin/python scripts/eval_shapes.py --shape all           # M8 shape boundaries
.venv/bin/python scripts/eval_materials.py --level all        # M8b friction/stiffness
```

Tip: on a machine that is also running the IDE, launch sweeps with `nice -n 10`
and `--workers 6` — 14 workers can starve the VSCode server and drop the session.

Outputs land in `results/` (CSVs = trial dataset, JSON = fits/reports, PNG = figures).
A trial's outcome labels: `held`, `drift` (>2 mm in the tip frame), `drop`, `no_grasp`;
incipient-slip ground truth = sustained tangential sliding >5 mm/s (kinematic, from sim
state). The vibration detector is scored *against* that truth (accuracy + latency), at
the ideal rate and downsampled to the TacTip camera rate (100 Hz default).

## Package layout

```
pinchlab/
  params.py     all tunables (ModelParams, BoxSpec, ControlParams, TrialSpec, Posture)
  model.py      URDF load/patch, scene build, HandKinematics (manifold solver, feasibility)
  tactile.py    per-tip force/centroid extraction + grasp-frame decomposition
  control.py    GripController: state machine, posture PD + gravity ff, f1 force loop,
                DIP spring, anti-gravity spawn hold, setpoint ramp, Phase B trajectory
                hooks, online vibration reflex
  trial.py      one episode: spawn→close→grip→release→hold, logging, slip truth, labels
  sweep.py      condition grid on the grasp manifold, anchor ladder + log bisection,
                parallel spawn pool, resumable CSVs
  slipdetect.py band-energy detector + scoring vs kinematic truth
  fit.py        μ_eff boundary fit, regime classification, basis regression,
                equation export
  plots.py      trial/map/boundary/slip figures
```

(`scripts/eval_shapes.py` adds the M8 shape boundaries; object shapes are selected
via `BoxSpec.shape`: box / prism / cylinder / disc / disc_edge / sphere.)

`pinch_sim.py` is the original single-file demo, kept as reference.