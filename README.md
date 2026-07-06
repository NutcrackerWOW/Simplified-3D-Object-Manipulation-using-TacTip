# pinchlab — 3D Object Manipulation, Pinching and Slipping Using Tactile Sensors

Drake simulation study on the Touch_Finger 2-finger hand with TacTip-style compliant
hydroelastic fingertips. Pipeline: force-controlled pinch trials → minimal-stable-force
sweep across grasp postures → fitted stability boundary → closed-form grip-force equation →
slip detection from force vibration → moving-grasp validation.

## Paper

*Rolling-Governed Pinch Stability: A Validated Minimal Grip-Force Rule for Soft
Tactile Fingertips* (in preparation for IEEE RA-L). LaTeX source: `paper/`
(`paper/main.tex` is the source of truth); markdown drafts `PAPER.md` (v1) and
`PAPER_v2.md` (v2); verified bibliography `refs.bib` + scan notes `references.md`.
Every number in the paper traces to an artifact in `results/`; the figures are
built by `scripts/make_paper_figs.py` into `results/paper_figs/`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # Drake 1.54, numpy, scipy, pandas, matplotlib
```

Tested on Python 3.12 (Linux/WSL2). The full sweep suite is CPU-parallel; see
"Running the pipeline" below for per-script worker flags.

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

    f1*(wave, pip, m) = (m·g/2) / μ_eff(wave, pip, m)
    μ_eff = c0 + c1·wave + c2·pip + c3·wave² + c4·pip² + c5·wave·pip + c6·(m − 0.045)

Coefficients, fit diagnostics, and validation live in `results/<tag>_fit.json` after
running the pipeline. The linear mass term models the measured decline of μ_eff with
mass (gravity torque worsens rolling); the fit is weighted by the audit-measured
per-mass seed noise (WLS).

### Measured results (sweepB, 120 conditions)

The posture map splits into **three regimes** — friction (fitted), support, and
fragile (both excluded, reported separately in the JSON):

- **Friction regime** (pip 0…10°, wave ±15°, 30–60 g; n = 60): the fit reaches
  R² = 0.60, RMSE = 0.054 over μ_eff 0.56–0.85 (unweighted metrics on WLS
  coefficients). μ_eff rises with PIP, is symmetric in wave tilt, and *declines with
  mass* (0.81/0.70 at 30/60 g, c6 ≈ −4.2 kg⁻¹) — rolling failure worsens under
  gravity torque, now an explicit model term.
- **Support regime** (pip ≈ −2°, mass-scaling classifier `classify_regimes`): the
  pads cradle the box; f* is mass-independent (0.15–0.31 N) — weight carried by
  geometry, not friction.
- **Fragile regime** (pip = −5°, audit-flagged): boundary existence is
  seed-dependent (bimodal) — only 2/5 independent seeds find a stable anchor, and
  those land at μ_eff above the Coulomb pair (cradling). Excluded from the fit as
  of the seed audit.

Additional envelope limits: **120 g exceeds the pinch capacity** at nearly all postures
(the hold boundary from below meets the ~2.5 N ejection boundary from above → empty
window). Held-out validation (`scripts/validate_equation.py`, 16 off-grid
in-envelope posture/mass points × 1.25×/0.6× probes × 2 seeds): **16/16 full hits**
— every point held at 1.25× and dropped at 0.6× on both seeds. Of the 3 deliberate
out-of-envelope probes, the support-band posture held below prediction and 120 g
dropped above it (both as characterized), while the 90 g probe now *passes* — the
explicit mass term fixed the extrapolation failure the old posture-only fit had
there (`results/sweepB_equation_validation.json`).

Slip detector (M6 → M6b): the original 6-trial split scored perfectly (3 TP / 3 TN,
latency ≈ 0.18 s) but cannot bound an error rate. The expanded study
(`scripts/eval_slip_big.py`: 90 episodes over 15 posture × mass conditions, 30 + 30
held-out test) exposed that a **single globally calibrated threshold fails** — the
pre-slip vibration floor varies 9× (ideal rate) to 72× (TacTip rate) across
conditions, so 3× the max training floor misses every slip (0 TP / 30 FN). The slip
burst is always separable from the *same condition's* floor (≥ 3.6×, median 12×
ideal / 45× TacTip), and two **deployable** threshold schemes fix it
(`scripts/rescore_slip.py` → `results/slip_eval_big_rescored.json`, offline from
cached episodes):

- **Scheduled threshold (primary):** log₁₀(floor) fitted over (wave, pip, mass)
  from the training floors — the same recipe as the grip-force map — then
  threshold = 3× the predicted floor at the current posture/load. Test score
  **30 TP / 1 FP / 0 FN / 29 TN at both rates** (matches the oracle-best global
  threshold), mean latency 0.17 s.
- **Adaptive threshold (supplement, model-free):** k× a causal running floor
  (trailing 2 s block-percentile with a 0.3 s guard gap), plus an absolute
  sensor-noise term (3× the quietest training floor) that stops ratio blow-ups
  on near-silent signals, 50 ms sustained-exceedance debounce, armed after grasp
  transients settle. Test score **30 TP / 0 FP / 0 FN / 30 TN at both rates**,
  mean latency 0.16 s — needs no posture model at all.

(Per-condition calibration — thresholds from each condition's own training
episodes — scores 28/0/2/30 and remains the diagnostic upper bound, but assumes
training data at every operating condition.)

Sweep seed-noise audit (`scripts/run_audit.py`, 12 conditions × 5 independent
single-seed bisections): in the friction regime, seed-to-seed f* spread is 2–7 % of
the median at 30 g (≈ the bisection notch) but grows to 7–32 % at 60 g — at higher
mass, f* uncertainty is seed-limited, not resolution-limited (consistent with the
fit RMSE). Near the regime boundary the boundary itself is seed-dependent: at
pip = −5° only 2/5 seeds find any stable anchor and those land at f* ≈ 0.09–0.14 N
(μ_eff 1.6–2.2, *above* the Coulomb pair — pure cradling), vs the 2-seed sweep's
0.34–0.62 N. Both findings feed back into the fit: pip = −5° is excluded as the
fragile regime, and the fit is seed-noise-weighted (see "Measured results").

### Phase B: the boundary is a 3-second boundary (rolling creep)

Carrying the 50 g box through wave ±10° @ 0.15 Hz at margin 1.3 over the fitted f*
drops the box — and so does a **motionless** hold at the same force. With 10 seeds
per condition (`scripts/phaseB_seeds.py` → `results/phaseB_seeds.json`): margin 1.3
fails **20/20** runs, failure time 5.3–11.1 s (motionless mean 7.8 ± 1.7 s, moving
mean 7.6 ± 1.6 s — statistically indistinguishable, so the carry motion is not what
kills the grasp). The bisection boundary (3 s hold window) is therefore
**duration-dependent**: sub-margin squeeze lets the box roll off by slow creep, with
utilization ρ ≈ 0.5 giving no warning and the vibration reflex firing too late for
abrupt roll-off. At **margin 2.0** (0.75 N) all three controllers (fixed / scheduled
/ scheduled+reflex) hold **30/30** (10 seeds × 3 modes) through the full wave sweep
with ≤ 0.85 mm drift (`results/phaseB_summary.json`, envelope probes in
`results/phaseB_envelope.json`, margin-1.3 failures archived as
`phaseB_*_margin1p3.*`). Since the earliest observed creep failure is 5.3 s, the
1.3 margin is only trustworthy for holds of ~4 s or less.

### Shape study (M8): μ_eff tracks rolling freedom, not friction

Same mass (50 g), same friction pair (μ 1.0/0.8), same 25 mm grasp width, reference
posture — only geometry differs (`scripts/eval_shapes.py`, `results/shape_*.json`):

| shape                            | contact                | f* (N) | μ_eff |
|----------------------------------|------------------------|--------|-------|
| disc (axis ∥ pinch axis)         | flat round faces       | 0.298  | 0.823 |
| box 25×25×12                     | flat square faces      | 0.435  | 0.564 |
| prism 25×50×12 (2× lever arm)    | flat square faces      | 0.435  | 0.564 |
| cylinder (vertical axis)         | curved rim (line)      | 0.464  | 0.529 |
| disc_edge (key pinch, upright)   | thin rim, gravity-roll | 0.572  | 0.429 |
| sphere d=25                      | point                  | 0.596  | 0.411 |

(Values from the 5-seed / 7-bisection-step rerun — every probe must hold on all 5
spawn seeds; notch resolution ≈ 5 %.) Coulomb friction is identical across rows, yet
f* spans 2× and orders exactly by the object's freedom to roll in the grasp — direct
evidence that the stability boundary is **rolling-governed**, the paper's central
claim. Box and prism land on *exactly* the same boundary notch even at this
resolution: doubling the lever arm about the grasp axis does not change f*, i.e. the
flat-face boundary is set by the contact patch, not the object's inertia about the
grasp axis. The `disc_edge` row is the lateral-prehension (key-pinch) configuration:
the disc stands like a wheel and gravity torque acts directly along the free rolling
axis. (The box value here is the 50 g probe — the sweep-fitted μ_eff at this posture
is 0.65 because μ_eff declines with mass.)

### Material study (M8b): friction scaling, window collapse, and a stiffness surprise

Box, 50 g, reference posture, one attribute varied at a time
(`scripts/eval_materials.py`, `results/material_*.json`; pair μ = harmonic mean of
box and tip values):

| level      | box μs/μd | pair μd | tip E (kPa) | f* (N) | μ_eff |
|------------|-----------|---------|-------------|--------|-------|
| fric_low   | 0.4/0.3   | 0.45    | 50          | — none | —     |
| fric_base  | 1.0/0.8   | 0.85    | 50          | 0.435  | 0.564 |
| fric_high  | 1.5/1.2   | 1.03    | 50          | 0.311  | 0.789 |
| tip_soft   | 1.0/0.8   | 0.85    | 25          | 0.504  | 0.487 |
| tip_stiff  | 1.0/0.8   | 0.85    | 100         | 0.360  | 0.681 |

(5-seed / 7-bisection-step values.) Three findings: (1) in the mid-to-high friction
range f* scales roughly like Coulomb (∝ 1/μ) but with a **constant rolling
discount** — μ_eff/μ_pair ≈ 0.66–0.77, i.e. rolling binds ~25–35 % below the sliding
limit; (2) at low friction the stability window **collapses entirely** — the hold
boundary rises (roll-off at 0.74 N) while the ejection boundary falls (squirt-out at
≥ 1.5 N), leaving no stable force at 50 g; (3) **softer fingertips are worse, not
better** (μ_eff 0.49 → 0.56 → 0.68 across 25/50/100 kPa): the naive bigger-patch
argument fails — a compliant dome yields as the object rotates, offering less
restoring moment against rolling.

## Running the pipeline

```bash
.venv/bin/python scripts/run_trial.py --force 0.6 --meshcat   # one visual trial
.venv/bin/python scripts/run_sweep.py --tag sweepB            # Phase A sweep (parallel, resumable)
.venv/bin/python scripts/fit_map.py --tag sweepB              # μ_eff map + equation + figures
.venv/bin/python scripts/eval_slip.py                         # slip detector: ideal + TacTip 100 Hz
.venv/bin/python scripts/validate_equation.py --tag sweepB --workers 10  # held-out test (16+3 points)
.venv/bin/python scripts/validate_moving.py --tag sweepB --margin 2.0  # Phase B carry
.venv/bin/python scripts/eval_shapes.py --shape all --seeds 5 --n-bisect 7 --workers 5   # M8 shapes
.venv/bin/python scripts/eval_materials.py --level all --seeds 5 --n-bisect 7 --workers 5 # M8b materials
.venv/bin/python scripts/eval_slip_big.py --workers 10        # M6b expanded slip study (90 episodes)
.venv/bin/python scripts/rescore_slip.py                      # offline re-scoring from cached episodes
.venv/bin/python scripts/phaseB_seeds.py --workers 10         # Phase B, 10 seeds per condition
.venv/bin/python scripts/run_audit.py --workers 10            # sweep seed-noise audit
bash scripts/run_rigor.sh                                     # all four studies, sequenced at 10 workers
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

## License

- **Code** (`pinchlab/`, `scripts/`, `bin/`, and other source files):
  [BSD 3-Clause](LICENSE).
- **Hand model** (`model/` — the Touch_Finger URDF and meshes, the author's own
  design), **datasets** (`results/` CSV/JSON/NPZ), and **paper figures**
  (`results/paper_figs/`):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — reuse with
  attribution (cite the paper above).