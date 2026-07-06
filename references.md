# Literature scan — Rolling-Governed Pinch Stability (RA-L)

Built 2026-07-06 from scratch. Every entry in `refs.bib` had its title, venue,
and year verified by web search this session; none were recalled from memory.
This file records (1) how the cited set maps to §II, (2) papers deliberately
**considered but not cited**, and (3) what §II must newly engage with given the
2022–2026 novelty scan.

---

## 1. Cited set → §II bracket map (16 entries in refs.bib)

RA-L counts references against the 6 (+2 over-length) page cap, so the set is
kept lean. The four §II brackets in the PAPER.md placeholder are filled as:

| §II bracket | Cited works |
|---|---|
| **(a) Soft-finger contact / limit surfaces** | `goyal1991planar`, `xydas1999modeling`, `kao1992quasistatic`, `kappassov2015tactile` (survey, framing) |
| **(b) Grip-force control & safety margins** | `johansson1984roles` (human precision-grip margin — the conceptual root of our ×2.0 rule), `howe1989sensing` (early robotic slip sensing) |
| **(c) Optical-tactile sensing & slip detection** | `chorley2009development` (TacTip origin — our fingertip lineage), `wardcherrier2018tactip`, `james2018slip` (learned TacTip slip baseline we contrast against a threshold rule), `yuan2017gelsight`, `johnson2009retrographic` (GelSight lineage, contrast to marker-based) |
| **(d) Compliant / hydroelastic contact simulation** | `elandt2019pressure`, `masterjohn2022velocity`, `castro2023unconstrained` (the exact PFC→velocity-level→SAP stack our Drake 1.54 testbed runs on) |
| **(e/f) Recent tactile grip / sim-to-real** | `lin2022tactilegym2` (TacTip in sim + sim-to-real; supports §VIII sim-only caveat and situates our work against learned tactile control), `jang2024soft` (nearest recent neighbor — Coulomb-state *estimation* from tactile sensing; §II must draw the estimate-vs-prescribe distinction in one sentence) |

**Narrative arc for §II:** the classical limit-surface line (a) predicts pair
Coulomb friction governs a fingertip grasp; the human-factors line (b)
establishes that a *margin* over the slip point is the control variable, not the
slip point itself; the tactile-sensing line (c) is where TacTip/GelSight fingers
came from and where slip is currently detected by *learning*, not a scheduled
threshold; the simulation line (d) is the compliant-contact machinery that lets
us measure a rolling-governed boundary at all. Our contribution sits in the gap:
a **measured** effective-friction map + a **validated** minimal grip-force rule
showing pair Coulomb friction *systematically overestimates* what the pinch can
hold, because the soft dome lets the object **roll** off below the sliding limit.

---

## 2. Considered, NOT cited (with one-line reason)

- **Johnson & Adelson, "Retrographic sensing" (CVPR 2009)** — kept (`johnson2009retrographic`) only as the GelSight root; if page pressure bites, drop in favor of `yuan2017gelsight` alone.
- **Heyneman & Cutkosky, "Slip classification for dynamic tactile array sensors" (IJRR 2016)** — strong slip-classification work but array/accelerometer-based, not optical; `james2018slip` is the closer baseline. Hold in reserve.
- ~~**"Soft Finger Grasp Force and Contact State Estimation from Tactile Sensors" (arXiv 2410.19684, 2024)**~~ — **promoted to cited** (`jang2024soft`, Jang, Bae & Haninger) 2026-07-06: nearest recent claim to ours; §II makes the contrast explicit (they *estimate* Coulomb state from tactile data; we *prescribe* a reduced effective-friction force rule).
- **"Bioinspired trajectory modulation for effective slip control" (Nature Machine Intelligence 2025)** — proactive (trajectory) vs reactive (grip-force) slip control. Interesting for §VI framing but out of scope for a sim-only minimal-force rule; not cited to avoid scope creep.
- **"Compliant In-hand Rolling Manipulation Using Tactile Sensing" (arXiv 2603.04301, 2026)** — rolling is *exploited* for manipulation, not treated as a *failure mode* to be bounded; different sign of the same phenomenon. Note as related but do not cite unless a reviewer raises rolling-as-feature.
- **"Planar Friction Modelling with LuGre Dynamics and Limit Surfaces" (2023)** — richer friction dynamics than we model; our claim is a static/quasistatic boundary. Out of scope.
- **Tactile Gym 1.0 / DIGIT / DigiTac papers** — subsumed by `lin2022tactilegym2` for our single sim-to-real citation.

---

## 3. What §II must newly engage with (novelty-claim protection)

The 2022–2026 scan (rolling/torsional grasp failure + tactile grip-force
minimization + hydroelastic-sim grip laws) found **no** published
rolling-governed *minimal grip-force rule* in compliant/hydroelastic
simulation. The nearest neighbors, and how our narrowed claim survives each:

1. **arXiv 2410.19684 (2024), soft-finger Coulomb-state estimation** — closest.
   They test whether a simplified Coulomb model *predicts* slip state from
   tactile force estimates. We instead *measure* that the stable boundary sits
   25–35 % **below** pair Coulomb because of rolling, and give a rule for the
   minimum force. Different object (predict-from-sensor vs. prescribe-force);
   §II should cite it and state the distinction in one sentence.
2. **Grasp-stability / transition-detection papers (2024)** that list "excessive
   rolling" as a failure mode — they *name* rolling qualitatively; none give a
   quantitative effective-friction discount or a validated force law. Our
   contribution is the *number* (μ_eff/μ_pair ≈ 0.65–0.75 rolling discount) and
   its validation (16/16 held-out).
3. **Learned tactile slip controllers (Tactile Gym, diffusion policies)** —
   these are data-driven and require training; our scheduled-threshold detector
   and closed-form rule are the interpretable, no-training contrast.

**Recommended §II/Conclusion wording** (consistent with grilling decision Q6 —
drop "first characterization"): claim only a *measured effective-friction map*
and a *validated minimal grip-force rule* for soft tactile fingertips under
rolling-governed failure, positioned against — not over — the estimation and
learned-control lines above.

---

## 4. Open items before §II is rewritten in PAPER_v2.md

- [x] arXiv 2410.19684 contrast citation added (`jang2024soft`) → 16 entries,
      per recommendation; authors verified via arXiv abs page.
- [ ] Decide whether `johnson2009retrographic` survives page pressure.
- [ ] Page/DOI fields marked from search summaries (Chorley ICAR 2009 has no
      DOI; Howe ICRA 1989 page range 145–150 inferred; Masterjohn DOI string
      10.1109/LRA.2022.3203210 inferred from Xplore doc 9874987) — re-verify
      against the publisher PDF at LaTeX-compile time.
- [x] Audit 2026-07-06: Tactile Gym 2.0 author list corrected to the 4-author
      list (Lin, Lloyd, Church, Lepora) per Bristol PURE + arXiv 2207.10763;
      Masterjohn/Yuan/Lin volume-issue-pages confirmed.
