# SCovox — Experiments Plan

The single canonical plan for evaluating SCovox: a low-memory, CPU-only,
metric-semantic RGB-D + LiDAR mapping system with low-bandwidth distributed
fusion. **Numbers live in [RESULTS.md](RESULTS.md); this file is the *what* and
*why*.** The three original plan documents (full-eval, uncertainty, feasibility)
are preserved verbatim under [archive/](archive/) — consult them for exhaustive
sub-experiment detail.

Status legend: ✅ done · 🔧 partial · ⬜ to do · 🚫 blocked

---

## 1. Thesis — six falsifiable claims

| ID | Claim | Evidence |
|---|---|---|
| **C1** | *Metric* accuracy — **core:** probabilistic Beta-occupancy is geometrically faithful on both modalities; **module:** the optional TSDF layer adds sub-voxel surface fidelity, evaluated for parity with TSDF baselines | E1 |
| **C2** | *Semantic* accuracy — competitive mIoU under matched inputs at the truncated K_top=2 state. *"Competitive" is a comparative claim: own-numbers characterise, but a baseline column (§2) is required to establish it.* | E2 · S1 |
| **C3** | Uncertainty is calibrated and decomposable into aleatoric/epistemic (the representational differentiator) | E3 · S1 |
| **C4** | *Low memory* — O(K_top) per-voxel state, small measured grid, favourable scaling | E4 · S1 |
| **C5** | *CPU-only, real-time-capable* — single-thread integration meets sensor rates; feasible on embedded ARM | E5, E7 |
| **C6** | *Low-bandwidth distributed fusion* — combined maps approach a centralised reference at small, **measured** payloads, with **loss- and outage-tolerance via snapshot resynchronisation**. *Bandwidth is a demand-side claim (measure and declare, E6.2); tolerance is a supply-side claim (demonstrate, E6.3/E6.4 — two cells, not a factorial).* | E6 · S2 |

**Central hypothesis (the thesis behind the claims).** C2, C3, C4 and C6 share
one load-bearing hypothesis: the truncated per-voxel state — **K_top = 2 tracked
classes + an OTHER bucket** — retains approximately all the information the full
C-class state carries, and is what simultaneously buys the uncertainty (C3),
memory (C4), and wire-bandwidth (C6) numbers. The substrate was designed to test
exactly this. K_top is therefore the thesis's *independent variable*, not one
ablation among many; it is proven or falsified by the **S campaigns** (§3): S1
measures all four legs against the same K_top sweep, S2 stresses the hypothesis
where it is hardest — the cross-robot fold, where `mergeDir` re-truncates the
union of the sources' top-K sets and a class dumped to OTHER cannot climb back.
Sufficiency is an **internal** comparison (the own full-K run is the reference)
— it does not wait on SLIM-VDB; only *competitiveness* does.

**Architecture stance (governs the whole matrix).** All probabilistic and
uncertainty claims rest on the conjugate **Beta–Dirichlet state alone**. The
TSDF layer is an *optional surface module* — for mesh output and baseline
parity — and carries **no uncertainty**: its fusion weight is not calibrated
confidence, and its exclusion from E3 is by design, not omission.
Consequences: E3 is TSDF-free; E4/E5 headlines are **core-state** numbers with
the TSDF grid costed separately; E1 reports what the module *buys* (surface
delta) beside what it *costs* (E4/E5 module lines) — the **costs-vs-buys
table** (three-row spec in E1; home: RESULTS.md Part II), the direct evidence
that TSDF is optional.

---

## 2. Datasets, protocol, references

**Principal protocol.** SemanticKITTI 06–10, first 100 scans, 10 cm, PolarNet soft
probabilities, 19-class taxonomy · SceneNet 13 validation trajectories × 300
frames, 5 cm, oracle labels, 14 classes. Principal params: w_occ=6, w_free=1,
κ₀=2, α₀=0.01, S_max=1000, **K_top=2**.

**Long-run protocol** (scaling only): KITTI-08 × 1000 scans; one SceneNet
trajectory ×3 concatenated (900 frames). Used by E4/E5 scaling cells.

**References.**
- *Occupied* — aggregated labelled points, majority class per voxel.
- *Free* — voxels traversed by ≥ m GT-pose rays with zero endpoint hits (m=3;
  sensitivity m∈{1,3,5}); never-observed voxels excluded. A visibility proxy →
  occupancy results are framed comparatively.
- *Surface (SceneNet, primary)* — reference mesh from TSDF fusion of GT depth at
  matched resolution. KITTI surface is secondary (no dense-depth GT).
- *KITTI dynamics* — geometric metrics reported twice: all classes, and excluding
  vehicle/person voxels.

**Evaluation principles.** Matched observations/poses/resolution/taxonomy for
every comparison. Paired statistics by sequence/trajectory (mean diff ± SE, 95%
CI). **n=3 per principal cell** (run-to-run SD reported separately from
across-unit SD). Dual-support scoring — *union* (pred∪gt) and *intersection*
(co-observed) — reported side by side. Directional hypotheses stated before
execution. Release builds, one executor thread, warm-up frames discarded.
**Matched-configuration rule (keyed per *comparison*, not per baseline):**
*surface* comparisons (H1.2) — both sides TSDF-on; *semantic/uncertainty*
comparisons (C2/C3) — config-irrelevant (the Beta–Dirichlet update path is
TSDF-independent; scored from the core cell); *memory/runtime* comparisons —
whole-system vs whole-system: SCovox **parity** (TSDF-on, own TSDF grid
counted) vs SLIM-VDB *with its SDF grid included* — the prior-round protocol
already measured SLIM-VDB this way (TSDF + semantic grids together) — and
SCovox **core** (TSDF-off) vs OctoMap. Net: no comparison carries a module
the other side lacks, in either direction; the corollary is that SCovox
legitimately appears at *two* configs in the same E4/E5 tables (stated there,
so it reads as matched reporting, not inconsistency).

**Baselines.** Two tiers. **OctoMap** (log-odds occupancy octree) is now *in scope*
as the C1/C4 geometric+memory baseline — apt-installable, CPU-only, no learned
weights, ingests the same clouds+poses; it fills at least one comparison column for
C1 (occupancy AUROC, H1.1) and C4 (bytes/leaf, H4.3) without waiting on anything.
**SLIM-VDB** (dense-Dirichlet OpenVDB) remains the semantic/uncertainty baseline for
C2/C3 but is **🚫 blocked** — source/image/weights absent from disk (see §7);
Voxblox/Kimera, S-BKI, segmenter-only stay out of current scope. All SCovox
own-numbers are unaffected; the SLIM-VDB *columns* fill in once it is obtained (or by
citing prior-round numbers — legitimate only if the protocol has not drifted, see the
encoding gate in §7).

---

## 3. Experiment matrix

### S — Sufficiency spine · the central hypothesis (C2·C3·C4·C6)

Two campaigns, both SCovox-internal and unblocked. Existing cells E2.2 and the
E3.4 full-K run are **absorbed here**, not run separately.

- **S1 — the K_top sweep, all four legs at once** ⬜. One paired campaign at
  K_top∈{1,2,3,full} (full = C: 19 KITTI / 14 SceneNet), core config off the
  frozen manifest with K_top as the swept field; every leg reported against the
  same sweep:
  ① *semantic* — dual-support mIoU (absorbs E2.2; scores H2.3);
  ② *uncertainty* — ECE, error-detection AUROC, sparsification vs the full-K
    reference, plus the p_OTHER·ln(C−K_top) coarsening bound checked against
    the observed per-voxel entropy gap (absorbs the E3.4 full-K run cell);
  ③ *memory* — the analytical 4+6·K_top B curve beside measured `[memSplit]`
    grid MB at each K;
  ④ *wire* — compressed bytes per transmitted voxel as f(K_top) through the
    E6.2 harness (extends E6.2 req. ③ from a single point to a curve).
  Per-frame time logged for free cross-feeds H5.2 (Dirichlet cost O(K_top)).
  n-policy: sweep cells n=1 descriptive; the thesis-bearing endpoints (K_top=2
  and full) n=3 paired.
  **Exposure statistic — pre-empts the cheapest attack** ("K=2 suffices because
  your data rarely shows >2 classes per voxel"). From the full-K run: the
  per-voxel distinct-observed-class count distribution, and the ①/② deltas
  re-reported *stratified on voxels with ≥3 observed classes* (class
  boundaries). Sufficiency must hold where it is actually tested.
  - **HS1.1** (= H2.3) K_top=2 within **0.005 mIoU** of full on every KITTI
    sequence across all runs. **HS1.2** |ΔECE| ≤ **0.01** and error-detection
    AUROC within **0.01** of full-K, both datasets. **HS1.3** the observed
    entropy gap ≤ the p_OTHER·ln(C−K_top) bound everywhere. **HS1.4** the
    ≥3-class stratum satisfies HS1.1/HS1.2 at **2×** tolerances
    (pre-registered; the stratum is smaller and noisier).
- **S2 — truncation at the fused fold, the hardest cell** ⬜. `mergeDir`
  re-truncates the union of the sources' top-K slots at every fold; a class
  dumped to OTHER cannot climb back — a pathology only a *fused* map with
  disagreeing sources can exhibit, which single-robot sweeps structurally
  cannot see. On the E6.1 *complementary* and *redundant* partitions, the 2×2
  {distributed, centralised} × {K_top=2, full}. **Truncation-at-merge loss ≡
  distributed(full) − distributed(K=2)**; centralised(full) − centralised(K=2)
  isolates truncation-without-fusion. This also states the centralised
  reference's config explicitly (both K cells), which E6.1 previously left
  open.
  - **HS2.1** truncation-at-merge mIoU loss ≤ **0.01** on both partitions.
    **HS2.2** overlap-region calibration (E3.6 machinery) degrades ≤ **0.01**
    ECE from centralised-full to distributed-K=2.
- **Standing evidence already in hand:** H3.1's confirmed nuance — *p_OTHER,
  not MI, is the out-of-top-K signal* — is direct evidence that OTHER carries
  the truncated mass in usable form; report it as thesis evidence, not a
  footnote (conditional on the † encoding gate, §7). A7 (OTHER ablation) shows
  OTHER, not the top-2 alone, is what buys it.

### E1 — Geometric (metric) accuracy · C1
- **Occupancy** — threshold-free AUROC + AP of p_occ vs occupied/free reference;
  at τ=0.5: precision/recall/F1/IoU, free-space FPR, {occ,free,unknown} confusion.
  ✅ both datasets (n=1).
- **Surface (SceneNet) — two configs by design (§1 stance)** — accuracy
  (pred→ref), completeness (ref→pred), F-score at τ∈{2.5,5} cm vs a 1 cm
  GT-depth TSDF-fusion mesh, reported for **both** surface sources:
  (a) *core-only* — marching cubes on the Beta-occupancy iso-surface, TSDF off —
  ✅ n=1 (**a result, not a stopgap**: this is what the core state alone
  delivers); (b) *module* — the TSDF zero-crossing (**true-SDF path wired**;
  needs the `share_tsdf:=true` re-capture). The (b)−(a) surface delta, paired
  with the module's memory/CPU lines from E4/E5, populates the
  **costs-vs-buys table (row ①)**.
- KITTI secondary surface @{10,20} cm — ⬜ (likely infeasible: no dense-depth GT).
- **H1.1** SCovox occupancy AUROC ≥ log-odds OctoMap under matched carving — ⬜
  *buildable now* (OctoMap in scope, §2).
  **H1.2** endpoint labelling costs surface completeness vs band-labelling TSDF at
  matched truncation — 🚫 (needs a band-labelling TSDF baseline / SLIM-VDB).
- OctoMap occupancy column — ⬜ buildable; SLIM-VDB columns — 🚫.
- **Costs-vs-buys table — named deliverable, fully specified. Home:
  RESULTS.md Part II, own "TSDF costs-vs-buys" subsection.** Three rows, all
  free from the paired capture (no extra runs):
  ① *surface gain* = E1(b) − E1(a) F-score at τ∈{2.5,5} cm;
  ② *memory cost* = the E4 `[memSplit]` TSDF-grid line (MB);
  ③ *runtime cost* = E5 per-frame time, TSDF-on − TSDF-off (H5.3).

### E2 — Semantic accuracy · C2
- **E2.1** principal comparison, dual-support, per-class IoU — ✅ scorer, n=1
  own-numbers (**KITTI cell conditional on evidence encoding †, §7**).
- **E2.2** K_top∈{1,2,3,full} sweep — ⬜ **absorbed into S1 ①** (run configs
  exist; endpoints at n=3).
- **E2.3** Dirichlet vs MV vs NAIVE on **full sets**, n=3 — 🔧 done n=1.
- **E2.4** soft vs argmax — 🔧 SceneNet done; **KITTI soft needs PolarSeg regen †** (§7).
- **E2.5** SLIM-VDB truncation sweep — 🚫.
- **H2.1** intersection-support narrows the SCovox–baseline gap vs union — 🚫
  (needs SLIM-VDB). **H2.2** Dirichlet ≥ NAIVE on every unit (SCovox-internal,
  buildable). **H2.3** K_top=2 within 0.005 mIoU of full on every KITTI sequence
  across all runs (SCovox-internal, buildable; = **HS1.1**).

### E3 — Uncertainty · C3  *(the "uncertainty round" — its own sub-matrix)*
The representational differentiator. Detailed design in
[archive/UNCERTAINTY_PLAN.md](archive/UNCERTAINTY_PLAN.md). Prerequisite code
(closed-form MI/expected-entropy exposure, snapshot dumper) is done offline in
Python — no scovox core change needed; the wire format already exports the raw
per-voxel sufficient statistics. **E3 is TSDF-free by design** (§1 stance): the
TSDF weight is not calibrated confidence and contributes no uncertainty signal —
all signals derive from the Beta–Dirichlet state.

| Sub | What | Status |
|---|---|---|
| E3.1 | Occupancy calibration + BALD decomposition (ECE/Brier/NLL, EIG epistemic, E_H aleatoric, sparsification/AUSE) | ✅ both datasets |
| E3.2 | Semantic calibration + error-detection + sparsification (top-label/classwise ECE, AUROC of {total, aleatoric, MI, vacuity, p_OTHER, MV-margin}) | ✅ KITTI **(conditional † — argmax vs soft, §7)** · ⬜ **SceneNet-soft pending** |
| E3.3 | **Decomposition validation** *(core scientific test)* — E3a evidence-scaling {25,50,100}% moves epistemic only; E3b label-noise ε∈{0,.1,.2,.4} moves aleatoric only | ⬜ **missing → build injectors** |
| E3.4 | Truncation fidelity + the p_OTHER·ln(C−K_top) coarsening bound vs full-K runs | ✅ selftest · ⬜ full-K run cell **→ S1 ②** |
| E3.5 | Downstream utility — informative frame selection; predicted vs realised gain | ⬜ P1 |
| E3.6 | Fusion + uncertainty — overlap-region calibration, peer-evidence γ∈{1,0.5} mitigation | ⬜ P1 |
| E3.7 | Recalibration transfer — one ESS temperature fitted on a held-out unit | ⬜ P1 |
- **H3.1** MI/vacuity beat total entropy at out-of-top-K detection; total/aleatoric
  win at misclassification (both confirmed on KITTI, with the nuance that *p_OTHER*,
  not MI, is the out-of-top-K signal). Baseline (SLIM-VDB alpha) columns — 🚫.

### E4 — Memory · C4
- **Analytical** per-voxel bytes vs K_top, vs dense 4C vectors, vs OctoMap — ✅.
  Split-build `DirVoxel` is **16 B** at K=2 (`4 + 6·K_top`: one `other` float +
  `cnt[2]` + `cls[2]`; static-asserted in `dir_voxel.hpp`) → **3.5–5.0×** vs dense
  4C (56 B SceneNet / 80 B KITTI). ⚠ **Smaller than the paper.** The paper's
  *unified* `SemDirVoxel` was **20 B** (`8 + 6·K_top`, two header floats) → 2.8–4.0×;
  the split refactor moved the free-mass float into `BetaVoxel::a_free`, so the
  current build *beats* the published table. **Part I "paper reproduction" will not
  reproduce the 20 B / 2.8–4.0× figures — it improves on them; state this, don't read
  it as a regression.**
- **Measured** grid MB + process RSS over time + allocated-voxel counts, n=3 — 🔧
  n=1; grid MB needs `log_mem_usage:=true` (`[memSplit]`). **Headline memory =
  Beta+Dirichlet core**; the TSDF grid is a separate `[memSplit]` line item
  (module cost → costs-vs-buys **row ②**), never folded into the core number.
- **Scaling** memory vs frames (long-run) and vs resolution {5,10,20} cm — ⬜.
- **H4.1** grid memory sub-linear in frames after coverage saturates. **H4.2** at
  matched allocation per-voxel advantage ≈ analytical **3.5–5.0×** (split-build 16 B
  struct; was 2.8–4.0× at the paper's 20 B `SemDirVoxel`). **H4.3** ≤
  OctoMap-with-semantics at matched resolution — ⬜ *buildable now* (OctoMap in scope,
  §2). **H4.2/H4.3 are core-state claims** (TSDF-off). Per the per-comparison
  rule (§2), **SCovox appears at both configs in the measured table** — core
  (TSDF-off) beside OctoMap; parity (TSDF-on, own TSDF grid counted) beside
  SLIM-VDB *with its SDF grid included*, as the prior-round protocol already
  measured it — matched reporting, stated so two SCovox numbers don't read as
  inconsistency. SLIM-VDB dense-Dirichlet column — 🚫.

### E5 — CPU runtime · C5
- Per-frame p50/p95/p99, RTF vs sensor rate (10 Hz KITTI / 25 Hz SceneNet),
  sustained Hz — ✅ n=1. **RTF ≡ wall-time per frame ÷ sensor period** (RTF < 1 =
  real-time); KITTI ≈ **0.97, TSDF off** — *at* the boundary, so H5.1 lives or dies
  inside run-to-run noise → **this is exactly what n=3 resolves** (report mean ± SD,
  not a single pass).
- **2-way** TSDF-vs-semantic stage split via `fused_walker:=false` — ✅ feasible,
  no code change.
- **5-way** carve/Beta/Dirichlet/TSDF/serialise split — ⬜ *buildable via a sampling
  profiler* (`perf record`, Release) **+ differential configs** (TSDF on/off,
  K_top 1 vs 6, `fused_walker` on/off): a decomposed cost model at zero instrumentation
  overhead. Only per-voxel `clock::now()` instrumentation stays rejected (it would
  dominate the very cost measured) — sampling sidesteps that.
- Scaling vs resolution / K_top / TSDF-on-off; CPU utilisation — ⬜.
- **H5.1** desktop single-core ≥ 10 Hz KITTI @10 cm, **TSDF off** (RTF ≤ 1; the
  margin is thin — gated on n=3). **H5.2** Dirichlet cost O(K_top), <15% of frame
  (2-way semantic fraction, cross-checked by the perf split). **H5.3** TSDF is the
  dominant optional cost — trade stated per configuration; its runtime line is
  costs-vs-buys **row ③** (§1 stance: core headlines are TSDF-off, the TSDF-on
  cell is the parity/module config).

### E6 — Distributed fusion · C6  *(largest remaining build)*
- **E6.1 Dependence-spectrum partitions** — three two-robot constructions:
  *disjoint* (coverage union) ✅ done; *complementary* (FOV left/right split =
  genuine independent evidence) ⬜; *redundant* (odd/even = double-count stress) ⬜.
  Metric: combined vs better-individual **and vs a centralised upper bound** →
  fraction of the centralised−individual gap recovered ⬜. The centralised bound
  runs at **both** K_top=2 and full — the full-K cell doubles as S2's reference.
- **E6.2 Bandwidth — measure and declare.** *The "low bandwidth" claim is
  demand-side: the delta publisher is a fixed-rate timer, so offered load is
  independent of link state — an unconstrained-link measurement fully characterises
  it. No traffic control needed.* Four requirements make the declared number
  reviewer-proof:
  1. per-publication compressed delta size **distribution** — mean, p95, max
     (bursts break links, not means);
  2. **rate-dependence** at {0.25, 1, 2} Hz — payload/s expected *sub-linear* in
     rate (deltas shrink at higher rates; a result in itself);
  3. measured wire **bytes per transmitted voxel** vs the dense C-vector
     alternative (the structural O(K_top) wire argument) — **S1 ④ extends this
     single point to the full K_top curve**;
  4. **transport overhead** — one interface byte-counter measurement
     (`ip -s link` / `/proc/net/dev`) with shared-memory transport disabled →
     report a DDS multiplier; else state "application payload, excluding
     transport overhead".
  Plus: LZ4 ratio · delta-vs-snapshot crossover · declared **QoS/queue depths** ·
  a **sustainable-envelope** note connecting the declared kbit/s to the previously
  observed 4 Hz subscriber-queue overflow. **kbit/s-per-robot headline.** — 🔧
  payload-sum only. *E6.2 characterises the current **ungated any-change**
  publisher — it doubles as E6.6's baseline row; once the gate exists, the
  fixed-rate "offered load is link-independent" framing above holds only for
  this baseline, and every declared bandwidth number states its stack level
  (raw / codec-6 / gated).*
- **E6.3 Consistency** — order-invariance, idempotence (dedup), late-join
  equivalence — ⬜ (cheap, high reviewer value). **Late-join does double duty as
  the E6.4 outage cell.**
- **E6.4 Robustness — collapsed to two cells, not a factorial.** *Tolerance is
  supply-side and follows largely by construction from evidence-summing +
  snapshot resync — demonstrate it, don't sweep it.*
  - **Loss cell** ⬜ — seeded ~10% delta drop via an application-layer lossy relay
    node (~50 lines; no privileges; reproducible by seed).
  - **Outage cell** ⬜ — the E6.3 late-join test *is* the 30 s blackout +
    snapshot-resynchronisation case.
  Together these support the qualified claim: "tolerates loss and outages via
  snapshot resynchronisation, demonstrated at 10% delta loss and a 30 s blackout."
  **Optional appendix (not claim-bearing):** the full {bandwidth × loss × delay}
  grid through the same relay; kernel netem/tc as a cross-check if passwordless
  sudo appears.
- **E6.5 Scaling** — N∈{2,3,4} partitions; aggregate bandwidth vs N — ⬜.
- **E6.6 Significance gate — the comms Part 2 policy experiment** ⬜ *(design:
  [comms_design_2026_07_30.md](../design/comms_design_2026_07_30.md) Part 2.
  The τ and κ arms already exist in `scovox_node` — `share_change_gate` keeps
  last-emitted shadow grids and gates on `share_gate_p_eps` (τ) OR relative
  evidence growth `share_gate_evidence_rel` (κ = 1 + rel); what the cell still
  needs code-side is the heartbeat arm, the two state-flip baselines, the τ(n)
  knob, and gate-state cost reporting)*. Send a voxel only when it matters:
  mean arm `|Δp_occ| > τ` **or** evidence arm `n > κ·n_last` **or** heartbeat —
  a decoupled surrogate for a per-voxel KL trigger (Trimpe & Campi decomposition).
  The merge-architecture resolution (design doc Q7) makes it lossy in
  *convergence rate only*: the gate changes when a replica attains a value,
  never what the fold produces from it. Four requirements:
  1. **Baselines, all three** — current any-change gate (= the E6.2
     measurement); OctoMap-equivalent state-flip gate; MARBLE-equivalent
     state-flip + binarized payload. The third is *the one to beat* on
     merged-map quality at equal bandwidth.
  2. **Sweep** τ∈{0.01, 0.02, 0.05, 0.1} × κ∈{1.5, 2, 4} + the heartbeat
     interval, on the same replay as the Part 1 codec measurement. *(Parameter
     map: τ = `share_gate_p_eps`, κ = 1 + `share_gate_evidence_rel` → sweep
     rel∈{0.5, 1, 3}; heartbeat = `share_heartbeat_sec`.)*
  3. **Metrics** — bandwidth; **peer lag** (time for a voxel's merged value to
     come within ε of the sender's — the quantity binarizing baselines cannot
     even express); merged-map agreement (SSMI) vs the ungated run; one
     planner-level metric (map quality only matters through its consumers).
  4. **Ablations** — negative information on/off (silence ⇒ `|Δp| ≤ τ`, a
     map-quality gain at zero bandwidth); τ fixed vs τ shrinking with last-sent
     evidence (the constant-KL correction — fixed τ over-transmits at thin
     evidence, waits too long at thick). Knobs: `share_gate_tau_ref_n` (0 =
     fixed τ) and `share_gate_tau_n_pow` — pow = 1 is the design doc's 1/n,
     pow = 0.5 is the exact constant-KL rate in the quadratic regime
     (KL ≈ n·Δp²/2p(1−p) ⇒ Δp* ∝ n^(−1/2)); the sweep decides.
  Stacks multiplicatively with codec revision 6 (Part 1: 0.28×) — the codec
  shrinks each record, the gate reduces how many exist; every bandwidth number
  declares its stack level (raw / codec-6 / gated).
  **Prerequisites from the design doc's open questions:** the code already
  carries a **v1 Dirichlet trigger** (any top-K slot change OR relative
  class-evidence growth) — either adopt it and declare it, or design the
  KL-derived trigger before the sweep; gate **state cost** is unmeasured (the
  shadow grids store the full last-emitted Beta/Dir voxel plus an 8 B emit
  timestamp ≈ 44 MB-scale at the measured capture, per-peer unless broadcast) —
  measure CPU + memory inside this cell (publish path, not carve path).
- **E6.7 Update-message size** ⬜ — *does the size of the ROS message itself
  affect delivery of updates?* Today one publish tick serializes **all** pending
  deltas into a single LZ4 `ScovoxMapBinary` message — snapshots reach MBs, and
  even steady-state deltas far exceed one 1500 B MTU frame, so DDS fragments
  every message. The topic is reliable/KeepLast(50), so fragment loss surfaces
  as retransmission latency and publisher-queue pressure, and the E6.4
  app-layer relay drops whole *messages* — message size is exactly the **blast
  radius of one drop** (one lost 2 MB snapshot ≫ one lost 20 KB chunk).
  Chunking pushes the other way: per-message overhead (header + pose envelope)
  and a worse LZ4 ratio (smaller compression windows). Sweep
  `share_max_voxels_per_msg` ∈ {unlimited, 64k, 16k, 4k, 1k} deltas/msg on
  (a) a clean link and (b) the E6.4 lossy relay at 10%; measure delivered-voxel
  fraction, total bytes on wire (chunking overhead + LZ4-ratio change), peer
  lag, and merged-map agreement. Chunking is sender-side only — the receiver
  merge is snapshot-replace per (source, coord), so chunk boundaries cannot
  change the converged state, only the path to it. Rides the E6.2 byte
  counters and the E6.4 relay; no new harness.
- **H6.4** fused-map mIoU loss < 0.01 at 10% delta loss (0.25 Hz publish); after
  the 30 s blackout, the late-joining robot's map equals the
  continuous-participation map within float tolerance after one snapshot resync.
- **H6.5** at matched bandwidth the significance gate beats the
  state-flip + binarized baseline on merged-map agreement; and outside
  heartbeat/loss windows the deterministic per-voxel bound
  `|p_recv − p_sender| ≤ τ` holds at every instant — the guarantee that
  justifies the OR form over an exact KL trigger, so it must be *verified*,
  not assumed.
- **H6.6** message size matters only under loss: at 10% relay loss, smaller
  messages raise delivered-voxel fraction and merged-map agreement (smaller
  blast radius per drop) until chunking overhead and the degraded LZ4 ratio
  dominate; on a clean link the curve is flat within noise. If the clean-link
  curve is *not* flat, that is a transport finding worth reporting on its own.

### E7 — Embedded feasibility · C5
Jetson Nano: sustained Hz, p95 latency, RSS, payload, stage breakdown on ARM;
full-state cross-platform determinism check. 🚫 **hardware-gated** (no Nano attached).

### A — Supporting ablations · P2  *(anchors, n=1, descriptive)*
κ₀/α₀ · S_max∈{250,1000,∞} · quality-weighting on/off · carve band · gate sweeps ·
A6 Hutter floor on/off (✅ done) · **A7 OTHER ablation** — re-score K_top=2
snapshots with p_OTHER dropped and the top-2 renormalised (offline, no runs):
the expected collapse of out-of-top-K AUROC shows OTHER, not the top-2 alone,
buys the uncertainty leg (feeds S1). Mostly run-level via existing params — ⬜
compute, not code.

---

## 4. What is deliberately *not* claimed
Scene completion (references are built from the same observations — map fidelity,
not completion of unseen space); SLAM (GT poses throughout; no drift/loop-closure);
GPU comparisons; multi-session persistence; **link-level performance under
arbitrary network conditions** (C6 claims measured demand + demonstrated loss/outage
tolerance, not a networking study — the {bw × loss × delay} grid is optional
appendix material).

---

## 5. Status at a glance

| Claim | Own-numbers | Blocking the full claim |
|---|---|---|
| C1 metric | ✅ E1 occupancy + core-only surface (n=1) | module (true-SDF) surface run · costs-vs-buys table · n=3 · OctoMap col (buildable, core config) · SLIM-VDB col 🚫 (parity config) |
| C2 semantic | ✅ dual-mIoU (n=1) | KITTI encoding † · rule sweeps at n=3 · K_top sweep → **S1** · SLIM-VDB col 🚫 (*external* "competitive" only — internal sufficiency lives in S) |
| C3 uncertainty | ✅ calibration/decomp; ⬜ E3.3 validation | **E3.3 injectors** · SceneNet-soft E3.2 · KITTI encoding † · SLIM-VDB col 🚫 |
| C4 memory | ✅ analytical (16 B, 3.5–5.0×); 🔧 measured (n=1) | grid MB (core/TSDF `[memSplit]` lines) · scaling · n=3 · OctoMap col (buildable, core config) |
| C5 CPU | ✅ RTF + 2-way split (n=1) | 5-way perf split (buildable) · scaling · n=3 · (E7 hardware) |
| C6 fusion | ✅ disjoint only | FOV/redundant · centralised bound (both K cells) · **E6.2 bandwidth spec ①–④** · consistency (late-join = outage cell) · **loss cell** · **E6.6 significance gate** (τ/κ arms in code; needs heartbeat + baselines + Dirichlet-trigger decision) · **E6.7 message-size sweep** |
| **S sufficiency (thesis)** | 🔧 E3.4 selftest ✅ · H3.1 p_OTHER signal ✅ (†) | **S1 four-leg K_top campaign** (incl. exposure statistic) · **S2 fused-fold 2×2** · A7 OTHER ablation |

---

## 6. Recommended next build order (all SCovox-side, unblocked)
0. **Freeze the capture manifest + pin the KITTI evidence encoding (both §7)** —
   prerequisites; the n=3 campaign is gated on both, and ② is **thesis-critical**,
   not hygiene: the uncertainty-sufficiency evidence (S1 ②, H3.1's p_OTHER
   nuance) is exactly what may shift under soft evidence. Do these before
   spending 3×.
1. **S1 sufficiency sweep** — the central-hypothesis campaign: four legs vs one
   K_top sweep, exposure statistic included (absorbs E2.2 and the E3.4 full-K
   cell; endpoints at n=3). A7 (OTHER ablation) is free offline compute on its
   snapshots.
2. **E3.3 decomposition validation** — the missing core scientific test for C3
   (frame-subsample + label-noise injectors; no runs blocked).
3. **OctoMap baseline** (apt-install) — cheap first comparison column for C1
   (H1.1 occupancy AUROC) and C4 (H4.3 bytes/leaf); unblocks "competitive" for those.
4. **E6 redesign + S2 fused-fold 2×2** — FOV/redundant partitions + centralised
   bound at **both K cells** + **E6.2 measure-and-declare (①–④, with ③ extended
   to the S1 K_top curve)** + consistency (late-join doubles as the outage
   cell) + **loss cell via the lossy relay**; the {bw × loss × delay} grid stays
   optional appendix.
5. **Module (true-SDF) E1 surface run** (`share_tsdf:=true`, parity cell) + **n=3
   fresh *paired* runs** off the frozen manifest (TSDF-on parity + TSDF-off core),
   retiring the n=1 caveat across E1/E2/E4/E5 and populating the costs-vs-buys
   table in one campaign.
6. **E3.2 SceneNet-soft** calibration (the one open Part III cell) + **E5 5-way
   perf split** (`perf record` + differential configs).
7. **E6.6 significance gate + E6.7 message-size sweep** — last because they are
   the steps needing mapping-side code, and both ride step 4's harness
   (partitions, relay, bandwidth counters). The τ/κ arms already exist
   (`share_change_gate` + shadow grids); the new code is the heartbeat arm, the
   two state-flip baselines, the τ(n) knob, gate-cost reporting, and the
   `share_max_voxels_per_msg` chunker. Decide the Dirichlet trigger first — the
   code's v1 (slot change OR class-evidence growth) is acceptable if declared.
   Codec revision 6 (comms Part 1) is engineering, not an experiment; it lands
   whenever, but bandwidth numbers must state whether it was active.

---

## 7. Blockers & open decisions

**True blockers (external dependency).**
- **SLIM-VDB baseline** 🚫 — absent from disk (no source, `slimvdb_docker` image,
  libtorch 2.1.2+cu121, NYUv2 weights, or prior `voxels.bin`). Scorers are ready and
  pure-Python. Options: user provides source/image · attempt upstream rebuild · cite
  prior-round numbers *(only if the protocol has not drifted — see the encoding gate
  below)* · keep deferred. Until resolved, the SLIM-VDB *columns* are blank (C2/C3
  "competitive", H1.2, H2.1 wait on it); SCovox own-numbers + the OctoMap column stand
  alone.
- **E7 embedded** 🚫 — needs a Jetson Nano.

**Were false blockers → now buildable.**
- **E6.4 network robustness** — *not* netem-gated **and no longer a factorial**:
  the claim needs exactly two cells — a seeded-loss run through an
  application-layer lossy relay (no sudo, reproducible) and the E6.3 late-join
  test as the outage case. The {bw × loss × delay} grid is optional appendix
  material; kernel netem/tc the optional cross-check if sudo appears.
- **E5 5-way stage split** — *not* unmeasurable: a sampling profiler (`perf record`,
  Release) + differential configs decomposes the cost at zero instrumentation
  overhead. Only the per-voxel `clock::now()` approach stays rejected.

**Gates on the n=3 campaign (resolve before spending 3× compute).**
- **① Capture manifest — freeze the run spec.** One canonical *paired* launch
  config (TSDF-on parity + TSDF-off core, §1 stance) so a single campaign feeds
  E1/E2/E4/E5 at once, rather than re-discovering a missing flag after the fact
  (the surface numbers already had to be re-captured because `share_tsdf` was
  off). E2/E3 are scored from the core cell — the Beta–Dirichlet update path is
  TSDF-independent by design, and the parity cell serves as the cross-check.
  Frozen fields in the manifest below.
- **② KITTI evidence encoding.** Principal protocol says PolarNet **soft** probs, but
  they are not on disk — so the ✅ KITTI cells (**E2.1, E2.4, E3.2**, marked **†**)
  most likely ran on **argmax/one-hot**. That is protocol drift, and not cosmetic:
  under one-hot evidence per-observation entropy is 0, so aleatoric arises only from
  cross-observation disagreement — the E3.2 conclusions (incl. the *p_OTHER*-vs-MI
  nuance) may shift under soft evidence. **Decide** — regenerate soft via PolarSeg, or
  formally accept argmax and restate the protocol; the **†** cells are *conditional*
  until then.
- **n=3 policy** — everything on disk is n=1 descriptive; headline paired stats need
  fresh triple runs, all off the frozen manifest.

**Capture manifest (frozen run spec — one *paired* run [TSDF-on parity +
TSDF-off core] → E1/E2/E4/E5):**

| Field | Value | Feeds |
|---|---|---|
| `share_tsdf` | `true` on the parity cell | E1 module (true-SDF) surface |
| `enable_tsdf` / `sdf_trunc` | **paired**: on / principal band (parity — E1 module surface, E5 TSDF-on, SLIM-VDB config) **and** off (core — H5.1 RTF, core memory, core-only surface, OctoMap config) | E1 both surfaces · E4 core/module · E5 |
| `log_mem_usage` | `true` (emits `[memSplit]`) | E4 grid MB |
| `fused_walker` | `false` on the timing run (2-way split) | E5 |
| resolution | 10 cm KITTI · 5 cm SceneNet | all |
| K_top | 2 (principal); S1 sweeps {1,2,3,full} off this same manifest | all · S1 |
| evidence encoding | **soft** (pending ②) — else log "argmax" explicitly | E2 · E3 |
| seeds | fixed, logged per run | n=3 reproducibility |
| snapshot checkpoints | on (offline E3/E4 re-score) | E3 · E4 |
| delta topic QoS / queue depth | declared + logged | E6.2 sustainable envelope |

---

## 8. Where things live
- **[RESULTS.md](RESULTS.md)** — all numbers (Part I paper reproduction · Part II
  full-eval E1/E2/E4/E5, **incl. the "TSDF costs-vs-buys" subsection (spec in
  E1)** · Part III uncertainty E1/E2). This is the results doc.
- **Harness** — `full_eval/` (E1/E2/E4/E5 scorers, `build_tables.py`),
  `uncertainty/` (E3 calibration/decomposition + `scovox_bin.py`), `fusion/`
  (two-robot), `kitti/` + `soft_scenenet/` + `scovox_eval/` (mIoU reproduction),
  loose `run_*.sh` / `batch_*.sh` drivers (SceneNet). See `full_eval/README.md`.
- **[results/](results/)** — per-experiment output tables and captures.
- **[archive/](archive/)** — the three original plan docs (full-eval, uncertainty,
  feasibility), superseded by this file but kept verbatim for detailed design.
- **paper_docs/** — upstream paper reference material (METHODS, design notes).