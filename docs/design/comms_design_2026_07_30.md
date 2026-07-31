# Comms design for the split Beta/Dirichlet map stream — wire compaction and the significance gate

**Date:** 2026-07-30 (measurements) · 2026-07-31 (prior-art survey folded in)
**Status:** Design doc — implementation not started. The measurements below are
real, captured from a live single-robot run (not estimates).
**Depends on:** `BinarySerializer` codec revision 5
([binary_serializer.hpp](src/scovox_core/include/scovox/binary_serializer.hpp)),
the `SemSplitMap` split substrate, `share_shaper`'s coalescing egress pacer, and
`dscovox_node`'s snapshot-replace ingest
**Wire format:** breaking — Part 1 proposes codec revision **6**, a new record
layout for the Beta and Dir streams. MAGIC, header framing, and the LZ4 envelope
([lz4_codec.hpp](src/scovox_core/include/scovox/lz4_codec.hpp)) are unchanged.
**Non-goals:** the TSDF stream layout, `K_TOP`, or any merge rule.

This document covers the two ways to cut `scovox_bin` bandwidth:

- **Part 1 — wire compaction.** How the bytes are encoded. Lossless or nearly
  so, no policy change, measured end-to-end. Ready to build.
- **Part 2 — the significance gate.** *Which* voxels are worth sending at all.
  Larger lever, changes observable behaviour, needs its own experiment. Also the
  only part of this design with a plausible novelty claim.

## Motivation

One robot sustains **4.0 MB/s (32 Mbit/s)** on its own `scovox_bin` at the 2 Hz
share rate. That is a lot for a field radio, and it scales with fleet size at the
merger. The question that started this was narrower — "would int instead of float
for the Beta/Dirichlet parameters save bandwidth?" — and the measured answer is
**no, not on its own: 7%**. The interesting result is where the bytes actually
are.

---

# Part 1 — Wire compaction (codec revision 6)

## What was measured

Captured 77 s (154 frames) of `/robot1/scovox_node/scovox_bin` from a live run:
`dscovox_single_robot.launch.py robot:=robot1`, 10 cm resolution, `share_tsdf`
off, semantics off, fed by an NDT-localized replay of
`2026_06_19_18_19_06__kalhan-map-test-2_`. The capture holds 38,872,057 Beta
records and 2,604,181 Dir records, 309.91 MB on the wire.

The frames were decoded offline and re-encoded several ways, then re-compressed
with the same raw-LZ4-block codec the wire uses. **Validation:** re-encoding the
frames unchanged and recompressing reproduces the observed on-wire total to
within 0.006% (309.89 MB computed vs 309.91 MB observed), so the variant numbers
are directly comparable to real link traffic.

## Results

| variant | raw | on wire (LZ4) | B/voxel | link @2 Hz |
|---|---|---|---|---|
| **V0 shipped** (f32, raw coords) | 1.00x | **1.00x** | 7.47 | 4025 kB/s |
| V1 u16 payloads, raw coords | 0.80x | **0.93x** | 6.96 | 3751 kB/s |
| V2 u8 payloads, raw coords | 0.71x | 0.92x | 6.84 | 3685 kB/s |
| V3 block coords + f32 | 0.42x | **0.42x** | 3.10 | 1670 kB/s |
| **V4 block coords + u16** | 0.22x | **0.28x** | 2.10 | 1131 kB/s |

Quantization fidelity at a 1/64 step, worst case over all 38.9M voxels:

| | mean `p_occ` error | p99 | max |
|---|---|---|---|
| u16 | 7.95e-07 | 3.08e-06 | **6.96e-04** |
| u8 | 7.68e-06 | 2.25e-04 | 1.99e-03 |

## Why int width alone is not the answer

Narrowing the payload floats to u16 shrinks the *raw* frame by 20%, exactly as
arithmetic predicts (a Beta record is 12 B of coord + 8 B of payload → 16 B). But
after LZ4 only 7% survives, because the compression ratio simultaneously drops
from 2.74x to 2.35x. LZ4 was already extracting that redundancy.

The evidence distribution shows why. Measured as evidence *above* the prior:

| field | p50 | p99 | max |
|---|---|---|---|
| `a_occ − 1` | **0.000** | 472.0 | 998.0 |
| `a_free − 1` | 72.0 | 872.0 | 998.7 |
| `other − (C−K)α₀` | 11.9 | 322.9 | 999.9 |
| `cnt[i] − α₀` | 0.000 | 0.000 | 0.000 |

Half of all Beta records sit at *exactly* the occupancy prior, and (in this
semantics-off run) every Dirichlet slot is exactly at prior. Those are long runs
of identical bytes — LZ4's best case. Removing bytes from an already
well-compressed region removes compressible material along with it.

Coordinates are the opposite: 12 of every 20 Beta bytes, high-entropy, and
essentially incompressible in the sender's arbitrary touched-set order. That is
where the bandwidth actually is.

## Design: codec revision 6

Two changes, both to the Beta and Dir record layouts only.

### 1. Block-run coordinate coding (the lossless 2.4x)

Voxels are grouped by their Bonxai leaf block. The grid runs `leaf_bits=3`, so a
block is 8×8×8 = 512 voxels — one block coord amortizes over up to 512 records.
Per block:

```
[bx: i32][by: i32][bz: i32]   // block coord = voxel coord >> 3
[mode: u8]
  mode 0: [bitmask: 64 B]              // 512 bits, bit = (lx<<6)|(ly<<3)|lz
  mode 1: [n: u16][idx: u16] × n       // in-block indices, same bit numbering
[payload] × n                          // in ascending bit/index order
```

`lx = x & 7`, etc. Mode is chosen per block by size: the 64 B bitmask wins at
n ≥ 32, the index list below that. Break-even is exactly 32 voxels, so both modes
are needed — ray-carved blocks are dense, surface blocks are sparse.

This is **fully lossless** — no quantization, no numerical invariant touched, no
accuracy claim to defend. On its own it is 0.42x (V3).

### 2. u16 payload quantization (0.42x → 0.28x)

Encode **evidence above the prior**, not the absolute α:

```
q = round((a − prior) / step)         clamped to [0, 65535]
a = prior + q · step                  (receiver)
```

Encoding relative to the prior is what makes this safe: an at-prior voxel maps to
exactly 0 and reconstructs bit-exactly, so `isPriorBeta`
([dscovox_node.cpp:622](src/scovox_mapping/src/dscovox_node.cpp#L622)), the
sender's emit gate, and `mergeBeta`'s prior-subtraction all keep working. The
occupancy priors are compile-time constants (`kBetaOccPrior` / `kBetaFreePrior`)
so they cost nothing on the wire; the Dir priors (`α₀` per slot, `(C − K_TOP)·α₀`
for OTHER) are already derivable from existing header fields.

**Step size must be carried, not hardcoded.** The representable range is bounded
by `evidence_saturation` (default 1000,
[map_interface.hpp:87](src/scovox_core/include/scovox/map_interface.hpp#L87)),
which is a runtime parameter. A hardcoded 1/64 step tops out at 1023.98 — it
happens to cover the default cap with 2.4% headroom, and silently saturates if
anyone raises the cap. Add one header field:

```
[quant_step: f32]    // 0.0 = payloads are f32 (unquantized)
```

The sender sets `quant_step = evidence_saturation / 65535`; with the default cap
that is 0.01526, essentially the 1/64 used in the measurement above. The sentinel
value 0.0 handles `evidence_saturation = 0` (cap disabled, so evidence is
unbounded and quantization is unsafe) by falling back to f32 payloads — the
receiver branches on the field rather than needing a separate format.

Record payloads become: Beta 4 B (`q_occ`, `q_free`); Dir 10 B (`q_other`,
`q_cnt[2]`, `cls[2]` — class ids stay u16 and are not quantized).

### Do not use u8

V2 buys one additional percent over V1 and costs an order of magnitude in
`p_occ` accuracy. It is also far too coarse for the Dir slots, where α₀ = 0.01.
Not worth the range risk.

### Compatibility

Same drill as the 4→5 prior bump: `FORMAT_VERSION` becomes 6, `deserialize`
rejects a mismatched revision, and a mixed-revision fleet fails loud with a
dropped frame rather than silently corrupting fused mass. That machinery already
exists. Both changes are confined to the record layout — the merge rules, the
priors, and the shaper are untouched.

## Rejected: delta-vs-last-sent coding

This was the strongest form of the original int idea, and it compresses best:
quantized values can be delta-coded where floats cannot, and 86% of records are
re-sends of an already-known voxel whose value moved slightly. Measured on the
Beta stream alone: 283.14 MB → 78.49 MB (**0.28x**), versus 0.38x for absolute
u16 — an extra 27%.

**It is rejected because it is incompatible with the architecture.** The merger's
ingest is snapshot-replace (`*v = d.data`,
[dscovox_node.cpp:462](src/scovox_mapping/src/dscovox_node.cpp#L462)), and
`share_shaper`'s entire losslessness argument rests on it: only the *latest*
value per source coord matters, so the shaper may coalesce away intermediate
values while pacing. Delta coding requires every intermediate value to arrive, in
order, per peer. Adopting it would:

- break the shaper's coalescing outright (it could no longer drop stale values),
- turn a dropped frame from "slightly stale" into "permanently corrupt", and
- require per-peer sender state plus an ack/resync path.

Not worth 27%. Revisit only if the topology ever becomes a reliable ordered
point-to-point link with no coalescing pacer in front of it.

Note that this rejection is what makes Part 2 attractive: additive pseudo-counts
are exactly what delta coding is not — a suppressed update is subsumed by the
next absolute send, so *skipping* is safe where *differencing* is not.

---

# Part 2 — The significance gate

## The measurement, and why its baseline is wrong

Across all 154 frames there were **zero** identical re-sends — the current change
gate never emits a byte-identical record twice. But 5,561,289 unique coords
generated 38,872,057 records: each voxel is re-sent about **7 times**, every time
with a genuinely changed value, as ray carving nudges `a_free` upward by small
increments (median inter-send delta is 256 LSB out of a 65535 range, and 46% of
deltas fit in 7 bits).

That 86%-of-traffic figure is real, but **it is measured against a baseline that
is weaker than OctoMap's default**, and the doc must not claim it as untapped
headroom without that caveat. OctoMap's `use_change_detection`
(`OccupancyOcTreeBase::updateNodeRecurs`) marks a voxel only when it is newly
created or when `occBefore != isNodeOccupied(node)` — i.e. when its classification
crosses p = 0.5. Pure log-odds accumulation on an already-free voxel is silently
suppressed, which is precisely the ray-carving traffic we are paying for. Our
"emit if the value changed at all" gate sends strictly more than stock OctoMap.

So the correct framing is: the state-flip gate is a **degenerate significance
gate** whose threshold is "crossed 0.5", and what follows is its graded
generalization. Benchmark against the state-flip gate, not against our own gate.

## What everyone else does, and what is actually novel here

A five-lens literature survey (multi-robot mapping; event-triggered estimation;
occupancy mapping; value-of-information and distributed fusion; deployed systems
and patents) converged on one answer: **the mechanism is not novel, the
application is.**

### Not novel — do not claim these

| Claim | Prior art |
|---|---|
| Threshold-gated ("deadband", "send-on-delta") transmission | Miśkowicz 2006; Åström & Bernhardsson 2002; OPC UA `AbsoluteDeadband`; BACnet COV; DIS dead reckoning (IEEE 1278, 1993) |
| Gating against the *last transmitted* value rather than the last updated one | OctoMap's `resetChangeDetection()`; OPC UA specifies "last cached value = last value pushed to the queue" |
| Per-voxel gating of volumetric map transmission | OctoMap change detection (2013); deployed in MARBLE (DARPA SubT) and FZI `vdb_mapping` |
| The Beta / Dirichlet voxel representation | BGKOctoMap (Doherty et al., ICRA 2017) for Beta occupancy; S-BKI (Gan et al., RA-L 2020) for the Dirichlet semantic counting model |
| "Most of our traffic is redundant re-sends" as motivation | The standard opening observation of the entire event-triggered estimation field |
| Gating a *Bayesian belief* on divergence from the last transmission | Battistelli et al. 2019 (KL trigger vs. a reference density); Marck & Sijs 2010 |

### Novel — gating instead of quantizing

Every deployed system that reduces map-sharing bandwidth does it by **destroying
the posterior on the receiving side**:

- **MARBLE** builds a diff tree carrying full `getLogOdds()`, then serializes with
  `binaryMapToMsg` — 1 bit per leaf. Reported ~100x reduction; the peer gets
  binary occupancy.
- **FZI `vdb_mapping`** offers an "overwrites" mode whose grid is
  `openvdb::Grid<Tree4<bool,…>>`, and its own README concedes the mode
  *"has the downside, that the probabilistic framework is lost on the remote
  side."*
- **US 11,312,379** (octree map sync) quantizes child nodes to a single bit.
- **Where2comm** (NeurIPS 2022) gates per BEV cell, but on absolute current-frame
  detection confidence with no temporal reference at all, and ships learned
  features that cannot be Bayes-fused.

None of them offers a bandwidth knob that preserves posterior correctness.
Because Beta pseudo-counts are **additive**, a suppressed update is subsumed
whole by the next absolute send: the gate is lossy in *convergence rate* only,
and exact in the limit. That is a different point in the design space from
quantizing, and it is the one that keeps multi-robot Bayesian fusion valid.

Four of the five survey lenses identified this property independently as the
defensible contribution.

### Novel — the evidence arm, stated carefully

Gating transmission on **confidence growth at constant probability** was not
found as a trigger in any of the five literatures. It is only expressible because
the substrate kept Beta pseudo-counts instead of collapsing to clamped log-odds,
and it fixes a real failure mode: a saturated voxel receiving confirming evidence
has Δ`p_occ` ≈ 0 and would be starved indefinitely by a pure Δp gate, so the peer
never learns the difference between "probably a wall" and "certainly a wall".

Two qualifications, both of which should be stated rather than glossed:

- The nearest relative in control theory is **the dual** of Trimpe & D'Andrea's
  variance-based trigger (TAC 2014), which fires when uncertainty has *grown*
  past an absolute bound under process noise. A static map has no process noise —
  variance only shrinks — so we fire when it has shrunk by a factor. The existing
  analysis does not cover that regime.
- **The quantity itself is not new**, only its use as a per-voxel transmission
  trigger. Rocha et al. 2005 (below) reduce their per-voxel information term to
  `I^l ≈ log(σ_l / σ'_l)`, a log-ratio of the voxel's uncertainty before and
  after an update. Our evidence arm fires when evidence doubles, i.e. when the
  Beta variance roughly halves, i.e. at `log(σ/σ') ≈ ½ ln 2`. That is the same
  measure. The difference is structural: theirs is a *term inside a sum* that is
  thresholded once per measurement, ours is the threshold itself, applied per
  voxel against the last value sent to the peer.

### Resolved: Rocha et al. 2005 is not prior art for the gate

**Rocha, Dias & Carvalho, "Cooperative multi-robot systems: A study of
vision-based 3-D mapping using information theory", *Robotics and Autonomous
Systems* 53(3–4):282–311, 2005** was flagged as the main novelty risk — an
entropy-based *information utility* deciding what map information is worth
sending teammates. Full text now in [docs/papers/rocha2005.md](docs/papers/rocha2005.md).
It is close, and it is not the same thing. Three separations, in order of
importance:

1. **It gates per measurement, not per voxel.** Eq. 64 defines
   `I_{k,i} = Σ_{l∈Z_{k,i}} I^l_{k,i}` — per-voxel mutual-information terms
   **summed** over every voxel a stereo measurement influences. Eq. 68 then
   thresholds that scalar at `I_min`, with a hard cap `s_kmax` on the number of
   measurements sent and ranking by utility under that budget. The per-voxel
   terms exist but no voxel is ever individually gated.
2. **It is memoryless with respect to the peer.** The utility is the information
   gain of a new observation against *the sender's own current map*. No
   last-transmitted state is tracked; the paper states the assumption plainly —
   "the sender robot assesses the measurement's utility by assuming that if the
   measurement is useful for itself it is equally useful for its teammates."
   This is the censoring-sensors pattern (*is this observation informative?*),
   not send-on-delta (*has the peer drifted from me?*). It structurally cannot
   express the latter.
3. **It shares sensor data, not map state.** The payload is `S_k = (x_k, U_k)` —
   a subset of range measurements plus the sensor pose — which each receiver
   re-integrates as if it had taken them itself. That sidesteps the problem this
   design addresses entirely: you cannot degrade a posterior you never transmit.
   It also does not apply to our architecture, where the merger ingests map
   state, and it makes every receiver repeat the integration cost.

Their voxel is also a Gaussian pdf over continuous *coverage* (the fraction of a
voxel occupied by matter), not a Beta over binary occupancy — so the trigger
arithmetic in this document does not carry over directly either.

Cite it as the closest conceptual ancestor for "don't send map information that
does not reduce teammates' uncertainty enough", and distinguish on (1) and (2).

### Remaining novelty risk

Unread: underwater 3D-grid map transmission over ~100 kbps acoustic links
(Łuczyński & Birk, OCEANS 2017 and follow-ons) — the most bandwidth-hostile
setting in the field, and the likeliest place an aggressive per-cell gate already
exists.

Patent posture, stated factually and with no infringement or validity assessment:
the closest granted-and-active claim is **US 12,236,779 B2** (Intel, collective
perception service), whose transmission-selection granularity is the *layer*, not
the cell, with a fraction-of-cells-changed threshold. The closest cell-granularity
disclosure is **US 2023/0110467 A1** (Intel, abandoned — prior art, not an
enforceable claim), which discloses differential cost-map transmission of cells
whose cost or confidence changed by more than a threshold versus previously
transmitted messages. Google Patents and USPTO were rate-limiting during the
survey; both should be re-verified before being relied on, and non-English
filings were not searched.

## Design sketch

### The trigger

Two arms, OR-combined, evaluated per voxel against the value **last transmitted**
for that coord:

```
send  iff  |p_occ − p_occ_last_sent| > τ            (mean arm)
       or  (a_occ + a_free) > κ · (a_occ + a_free)_last_sent    (evidence arm, κ ≈ 2)
       or  now − t_last_sent > T_heartbeat          (liveness arm, see below)
```

This is not two heuristics stapled together. Trimpe & Campi (EBCCSP 2015, Eq. 5)
show the KL event trigger on a Gaussian belief expands into exactly two terms — a
mean-shift term and a log-precision-ratio term. The mean arm is the first; the
evidence arm is the second. The rule is a **decoupled surrogate for a per-voxel
KL trigger on the Beta posterior**, and should be presented as such.

### Why keep the OR form rather than compute the Beta KL

Beta KL is closed-form (log-betas plus digammas) and cheap, so "KL is too
expensive" is not a defensible justification and should not appear in the writeup.
The real reason is that the decoupled form gives a **hard deterministic
guarantee** the KL trigger does not: `|p_recv − p_send| ≤ τ` for every voxel at
every instant. For a map that feeds a planner, a per-voxel bound is worth more
than a single scalar information budget.

The honest cost of that choice, and it should be stated: the two arms are in
mismatched units. The evidence arm is constant-KL — Beta differential entropy
drops by exactly ½ ln 2 = 0.5 bits per doubling, and
`KL(Beta(2a,2b) ‖ Beta(a,b)) → ½(ln 2 − ½) = 0.0966 nats` independent of p. The
mean arm is not: at p = 0.7, τ = 0.05, the KL between successive sends grows
0.063 nats at n = 8 → 0.42 at n = 64 → 3.29 at n = 512. A fixed τ therefore
over-transmits when evidence is thin and waits too long when it is thick. Making
τ shrink with n is the obvious fix and should be an ablation.

### Two things the trigger arithmetic implies

- **The evidence arm is log-odds quantization in the one-sided regime.** Doubling
  `(a_occ, a_free)` leaves log-odds `L = ln(a_occ/a_free)` *exactly* unchanged in
  general — the arms are orthogonal coordinates of the Beta family. But a voxel
  receiving only one-sided evidence keeps the other parameter pinned at the
  prior, so `L = ln(n − 1)` and doubling gives ΔL → ln 2 = 0.693 nats. Most
  voxels in a static scene are one-sided, so operationally the arm behaves as a
  uniform log-odds quantizer there. Justify it by the constant-KL property above,
  not by "evidence doubled".
- **A fixed τ on p is not a fixed threshold in log-odds.** `ΔL = τ/(p(1−p))`, so
  τ = 0.05 gives ΔL = 0.20 at p = 0.5 but 0.75 at p = 0.9 — a coarse gate near
  certainty, fine near ambiguity. That is a soft, graceful analogue of OctoMap
  clamping and is probably the behaviour we want, but it is a design decision to
  be argued rather than assumed.
- **Sanity bound on κ.** OctoMap's default `l_max = 3.5` corresponds to
  `a_occ ≈ 33`, about 5.1 doublings from prior. Past that point the evidence arm
  is spending bandwidth on information OctoMap deliberately discards. Our own
  `evidence_saturation` (default 1000) is ~10 doublings, so the arm fires ~10
  times per voxel over its lifetime — cheap, and self-sparsifying, since
  doublings are logarithmically spaced in observation count.

### Liveness: heartbeat / max suppression interval

Every deployed instance of this pattern has one — Årzén's event-based PID (1999),
PI's `ExcMax`, DIS dead reckoning, ETSI CPM's 1 s rule. It is not optional here
either, for a reason specific to our link: on a lossy radio, **"suppressed" and
"dropped" are indistinguishable to the receiver**. A bounded max-silence interval
per voxel (or, more practically, per block) converts an unbounded corruption
window into a bounded staleness window. Silberstein et al. (VLDB 2007) is the
direct treatment of the suppression-vs-loss ambiguity.

### Negative information (free, unexploited)

Under the gate, *silence itself carries information*: it tells the peer that
`|p_occ − p_last_received| ≤ τ`. Every serious version of this technique uses it
(Sijs/Noack/Hanebeck FUSION 2013; Olston et al.; Battistelli et al., who
reconstruct and deliberately flatten the suppressed posterior). Discarding it
biases the receiver toward the stale value. This is a map-quality improvement at
literally zero bandwidth and should be in the experiment.

### Adaptive τ (future work, and where a real contribution is)

A single fixed global τ is explicitly the naive baseline in Olston et al.'s
precision-bounded caching work (SIGMOD 2001/2003); their actual contribution is
non-uniform, adaptive allocation of per-object bounds to hit a global precision
target at minimum bandwidth. The analogue here — tighter τ near frontiers and
planned paths, looser in settled interior — is the most defensible place to make
a genuine contribution rather than an application. Out of scope for v1.

## Known correctness caveat: the censored Beta

Deterministic deadbands break conjugacy. After suppression the receiver's true
posterior for a voxel is a **censored** Beta, not a Beta, so treating the
last-received `(a_occ, a_free)` as the current belief is biased toward stale
(Han, Mo, Wu, Weerakkody, Sinopoli & Shi, TAC 2015, for the Gaussian case). Two
remedies exist and both are worked out only for Gaussian: negative-information
filtering (above), and stochastic triggers designed to preserve closed form.

For our substrate the bias is bounded and conservative in the safe direction —
the receiver's evidence count is never more than κ× stale, so its variance is at
most ~κ× the sender's, i.e. the peer is *less* confident than the truth, never
more. Worth stating explicitly since a planner consuming the merged map cares
about exactly that direction. Working out the Beta case properly is a real open
problem, and a good future-work paragraph if this is ever written up.

## Validation plan

The gate is a *policy* change — lossy in convergence rate rather than in value —
so it needs its own experiment, not just a bandwidth number.

1. **Baselines, all three:** current any-change gate; OctoMap-equivalent
   state-flip gate; MARBLE-equivalent state-flip + binarized payload. The third
   is the one to beat on map quality at equal bandwidth, and it is the comparison
   the deployed literature would ask for.
2. **Sweep** τ ∈ {0.01, 0.02, 0.05, 0.1} × κ ∈ {1.5, 2, 4}, plus the heartbeat
   interval, on the same replay used for Part 1.
3. **Metrics:** bandwidth; *peer lag* (time for a voxel's merged value to come
   within ε of the sender's — the quantity the binarizing systems cannot even
   measure); merged-map SSMI / agreement against the ungated run; and a planner-
   level metric, since map quality only matters through its consumers.
4. **Ablations:** with and without negative information; τ fixed vs τ scaled by
   1/n (the constant-KL correction).

Stacks multiplicatively with Part 1 — codec revision 6 shrinks each record, the
gate reduces how many records exist.

---

## Open questions

1. **Semantics were off in this capture** (`use_semantics:=false`), so every Dir
   `cnt` slot sat at prior. Dir is only 6% of records so it cannot move the
   totals much, but the Dir layout should be re-measured against a run with the
   seg pipeline live before revision 6 is frozen. The gate design also needs a
   Dirichlet trigger — the same KL decomposition applies, but it has not been
   worked out here.
2. **The first frame is a 3.5M-voxel reconnect snapshot** — 9% of all records in
   the capture. Block-run coding should help it disproportionately (a full-grid
   dump is maximally dense), which would improve reconnect behaviour behind the
   shaper. Not separately quantified.
3. **CPU cost is unmeasured.** Block grouping needs a sort by block key per frame;
   the gate needs per-voxel last-transmitted state (~8 B × 5.5M coords ≈ 44 MB,
   and it is per-peer if the fleet is not broadcast). Both land on the publish
   path, not the carve path.
4. Even at 0.28x the link sits near 1 MB/s. If a hard radio budget is the real
   constraint, Part 1 alone does not reach it and Part 2 is required.
5. ~~Obtain Rocha et al. 2005.~~ **Resolved 2026-07-31** — obtained, read, and
   distinguished; see *Resolved: Rocha et al. 2005 is not prior art for the
   gate*. The remaining unread item is the underwater acoustic map-transmission
   line.

## Reproducing the measurement

The three analysis scripts are checked in under
[scripts/wire_study/](scripts/wire_study/). They are read-only: they decode
captured frames and re-encode them offline, and touch no scovox source.

```bash
# with a run live (any robot publishing scovox_bin):
ros2 bag record -o /tmp/scovox_bin_cap /robot1/scovox_node/scovox_bin
# ^C after ~60 s, then:
python3 scripts/wire_study/wire_study.py       /tmp/scovox_bin_cap   # variant sizes
python3 scripts/wire_study/frame_redundancy.py /tmp/scovox_bin_cap   # re-send analysis
python3 scripts/wire_study/delta_study.py      /tmp/scovox_bin_cap   # delta coding
```

Requires `python3-lz4` and `numpy` in the ROS environment. `wire_study.py` prints
the V0 reproduction check — if its V0 LZ4 total does not match the capture's
actual on-wire bytes, the decoder has drifted from the codec and the other
numbers should not be trusted.

## References

Ordered by how load-bearing they are for Part 2.

**The trigger, canonically**
- M. Miśkowicz, "Send-On-Delta Concept: An Event-Based Data Reporting Strategy",
  *Sensors* 6(1):49–63, 2006. The standard citation for the rule.
- K.-E. Årzén, "A simple event-based PID controller", *IFAC World Congress*, 1999.
  Deadband plus max-period safety timer — the heartbeat precedent.
- K. J. Åström & B. Bernhardsson, "Comparison of Riemann and Lebesgue sampling
  for first order stochastic systems", *CDC*, 2002. Why gate at all.
- S. Trimpe & M. Campi, "On the Choice of the Event Trigger in Event-based
  Estimation", *EBCCSP*, 2015. The taxonomy; Eq. 5 is the two-term KL expansion
  this design's two arms instantiate.
- S. Trimpe & R. D'Andrea, "Event-Based State Estimation With Variance-Based
  Triggering", *IEEE TAC* 59(12):3266–3281, 2014. The dual of the evidence arm.

**Gating a Bayesian belief**
- G. Battistelli, L. Chisci, L. Gao & D. Selvi, "Event-triggered distributed Bayes
  filter", arXiv:1902.09825, 2019. Closest architectural match: KL against a
  reference density = the last transmission.
- A. Mitra, J. Richards, S. Bagchi & S. Sundaram, "Event-Triggered Distributed
  Inference", *CDC*, 2020. Relative/multiplicative belief trigger with a.s.
  consistency and an exponential rate — the proof template for "the gate does not
  break convergence".
- Y. Han, Y. Mo, J. Wu, S. Weerakkody, B. Sinopoli & L. Shi, "Stochastic
  Event-Triggered Sensor Schedule for Remote State Estimation", *IEEE TAC*
  60(10), 2015. Deterministic deadbands destroy conjugacy.
- J. Sijs, B. Noack & U. Hanebeck, "Event-based state estimation with negative
  information", *FUSION*, 2013.
- R. Rago, P. Willett & Y. Bar-Shalom, "Censoring sensors: a low-communication-rate
  scheme for distributed detection", *IEEE TAES* 32(2):554–568, 1996. The
  optimality argument for threshold-shaped gates.

**Volumetric mapping and deployed map sharing**
- A. Hornung, K. M. Wurm, M. Bennewitz, C. Stachniss & W. Burgard, "OctoMap: An
  efficient probabilistic 3D mapping framework based on octrees", *Autonomous
  Robots* 34(3):189–206, 2013. Change detection = the state-flip baseline;
  clamping policy.
- K. Doherty, J. Wang & B. Englot, "Bayesian Generalized Kernel Inference for
  Occupancy Map Prediction", *ICRA*, 2017. The Beta occupancy voxel.
- L. Gan, R. Zhang, J. Grizzle, R. Eustice & M. Ghaffari, "Bayesian Spatial Kernel
  Smoothing for Scalable Dense Semantic Mapping", *IEEE RA-L* 5(2), 2020. The
  Dirichlet semantic counting model.
- J. Ohradzansky et al., "Multi-Agent Autonomy: Advancements and Challenges in
  Subterranean Exploration", *Field Robotics*, 2022 (arXiv:2110.04390) and
  `github.com/dan-riley/marble_mapping`. Deployed diff-map sharing; binarized.
- FZI `vdb_mapping` / `vdb_mapping_ros2`. The direct architectural analogue, and
  the explicit statement that its low-bandwidth mode loses the posterior.
- V. Reijgwart, C. Cadena, R. Siegwart & L. Ott, "Efficient volumetric mapping of
  multi-scale environments using wavelet-based compression", *RSS*, 2023.
  Saturated-region skipping; coefficient thresholding.
- R. Rocha, J. Dias & A. Carvalho, "Cooperative multi-robot systems: A study of
  vision-based 3-D mapping using information theory", *Robotics and Autonomous
  Systems* 53(3–4):282–311, 2005. Full text at
  [docs/papers/rocha2005.md](docs/papers/rocha2005.md). The closest conceptual
  ancestor; per-measurement, peer-memoryless, shares measurements not map state.
  Eq. 64 (utility), Eq. 65 (`log σ/σ'` closed form), Eq. 68 (the gate).

**Bandwidth-aware sharing elsewhere**
- C. Olston, J. Jiang & J. Widom, "Adaptive Filters for Continuous Queries over
  Distributed Data Streams", *SIGMOD*, 2003; and C. Olston, B. T. Loo & J. Widom,
  "Adaptive Precision Setting for Cached Approximate Values", *SIGMOD*, 2001.
  The adaptive-τ precedent.
- A. Silberstein, G. Puggioni, A. Gelfand, K. Munagala & J. Yang, "Suppression and
  Failures in Sensor Networks: A Bayesian Approach", *VLDB*, 2007. Suppression vs.
  packet loss.
- Y. Hu et al., "Where2comm: Communication-Efficient Collaborative Perception via
  Spatial Confidence Maps", *NeurIPS*, 2022. Per-cell gating, but on absolute
  confidence with no temporal reference.
- ETSI TR 103 562, Collective Perception Service object-inclusion rules. A
  deployed per-object send-on-delta gate against last-transmitted state.
- OPC UA Part 4 §7.22.2 `DataChangeFilter` (`AbsoluteDeadband` / `PercentDeadband`);
  BACnet COV (ASHRAE 135); AVEVA/OSIsoft PI exception & compression. The
  industrial standards that make the mechanism unclaimable.
