# `scovox_node.cpp` — code review and update list — 2026-09-06

**Scope.** Read-only review of `src/scovox_mapping/src/scovox_node.cpp`
(3 421 lines, `SCovoxNode`) at commit `703eb7b`, working tree clean for this
file. The question asked was "is this code free of code smells, redundant
code, unnecessary complication?" The answer is no, and this document is the
list of what to change, ordered so the safe items come first. No code was
changed for this review. It does not re-open anything from
`docs/archive/code/code_review_2026_09_04.md`; where an item there overlaps (M1, M2,
L4), it is cross-referenced rather than repeated. Tier G, added later the
same day, carries that review's open items forward with their status
re-checked; the cross-references stand.

**Addendum 2026-09-06, same day.** Tier F re-verifies
`scovox_slot_rules/scovox_cleanup.md` (last edited 2026-09-04, `64e8aa3`)
against the tree at `77b99da`, which differs from `703eb7b` by one docs
commit; `scovox_node.cpp` is byte-identical between the two. That document is
a work list for the core deposit path, scoped to the SceneNN replay. Three
commits landed after most of it was written (`e316f07` batching default-on,
`1101e53` count storage + `quality` removed + `nhit` compiled out, `d5da6a8`
exact DDA), and they resolve, invert or moot most of its items. Each is graded
below; the ones still open are checkboxes like the rest of this list.

**Addendum 2026-09-06, later the same day.** Tier G re-verifies
`docs/archive/code/code_review_2026_09_04.md` (1 688 lines: H1–H4, M1–M11, L1–L7,
S1–S11, an errata section and six addenda through 2026-09-06) against
`77b99da`. Its S-series was fixed by the eleven commits `f202860..32121f2`
on 2026-09-05 and every fix is still in the tree; its H/M/L items marked
open are still open, with two exceptions noted in G3 and G4. One caveat on
line numbers: the edit described in G14 landed as `a062908` (2026-09-06
14:59). It adds 7 lines to `scovox_node.cpp` after `:839` and 9 more inside
`main()` (3 437 lines at HEAD). Every `scovox_node.cpp` pointer in this
document is a `77b99da` number (3 421 lines); add 7 to anything past `:839`
and 16 to anything inside `main()` when reading HEAD. The review itself was
moved to `docs/archive/code/` in the working tree afterwards (an uncommitted
`git mv` at the time of writing); it is cited at that path throughout.

**Addendum 2026-09-06, core change that reaches this node.**
`SemSplitMap::Params::sem_top_k` now defaults to `K_TOP` — ingest keeps only
the `K_TOP` most probable classes of an observation and the rest falls to
`other()`. The node does not name this field: it assigns into a
default-constructed `SP.semsplit` (`:116-136`), so it inherits the new default
rather than overriding it, and the ROS path is capped exactly as the replay is.
Two consequences for the items below. First, `semantic_topk_trunc` (`:425`,
default `0` = off) is no longer the only truncation on the ROS path; it now
only matters for a cap *tighter* than `K_TOP`, and setting it above `K_TOP`
cannot loosen the library's. Second, A3's `semantic_top_k` is still dead —
it writes the legacy `scovox::Params::top_k`, not `SemSplitMap::Params::
sem_top_k` — but deleting it is now a *naming* decision as well as a dead-code
one, because a reader who finds it will reasonably expect it to reach the cap
that exists. See §4.9 of `scovox_code_structure.md`.

**Addendum 2026-09-07, core change that reaches this node.** The walkers'
wall-clock brackets are now switchable: `SCOVOX_WALKER_TIMERS`
(`scovox_core/include/scovox/walker_timers.hpp`, **default 1** — the shipped
behaviour is unchanged) folds the ten `steady_clock::now()` reads in
`scovox_map_split.hpp` away at 0. Two consequences here. First, a new item
**G16**: at 0 the node's `tsdf_ms` / `sembeta_ms` fields read `0.00` with no
indication why, and their log lines are a frozen parser contract. Second,
**G15 is closed** as part of the same work — `buildSwitches()` now names all
ten live switches, so a `WALKER_TIMERS=0`, a margin-probe and a counters
build are each self-identifying in the startup log the node prints at
`:277`. Numbers and gates are in `scovox_slot_rules/REVIEW_LOG.md` E-W30 and
§4.12 of `scovox_code_structure.md`; none of them is in a comment.

**Method.** Every item was checked against the source at the cited
`file:line`; all line numbers are `scovox_node.cpp` numbers at HEAD
`77b99da` unless another file is named (see the second addendum for the
working-tree offset). Supporting reads:
`include/scovox/node_utils.hpp` (the shared helpers that already exist),
`include/scovox/topk_provider.hpp` (constructor and setters),
`scovox_core/include/scovox/map_interface.hpp` (`scovox::Params`),
`CMakeLists.txt` (what links what), `src/dscovox_node.cpp` (no duplicated
helpers found there), `test/` (no test instantiates `SCovoxNode`). For tier
F: `scovox_core/include/scovox/{beta_voxel,dir_voxel,sem_split_map,
scovox_map_split}.hpp`, `scovox_core/src/sem_split_map.cpp`,
`scovox_slot_rules/scovox_scenenn/src/replay_scenenn.cpp`,
`scovox_slot_rules/scripts/{build_rules.sh,slot_readout.py}`, the three RGB-D
yamls, `git log -S` for the removing commits. For tier G:
`docs/archive/code/code_review_2026_09_04.md` in full; `binary_serializer.hpp`
(block-run encoder and decoder), `consensus_merge.hpp`, `dscovox_node.cpp`,
`node_utils.hpp` (`argmaxClassConfidence`), `e0_counters.hpp`,
`version.cpp`, `map_lock.hpp`, both packages' `test/` directories,
`scripts/wire_study/wire_study.py`, `scovox_slot_rules/scripts` for `.md`
references, the three eval launch files for `w_occ` / `kappa0` /
`semantic_band_length`, and `git log` / `git status` / `git branch` for the
errata and the working-tree state. No build, test or replay was run.

**How to read the tiers.**

| tier | meaning | proof required before merge |
|---|---|---|
| A, B | behaviour-neutral: dead code and duplication | build + `./dev.sh ros-test` green |
| C | changes runtime behaviour on purpose | the item says what changes; decide first |
| D | structural: moves code, must not move bytes | wire frames from one bag byte-identical before/after (see §Verification) |
| E | comment hygiene under the standing rules | none beyond a diff read |
| F | items carried over from `scovox_cleanup.md`, re-verified | per item; most are closed on evidence |
| G | items carried over from `docs/archive/code/code_review_2026_09_04.md`, re-verified | per item; the S-series is closed on evidence, the rest are mostly decisions |

**Standing rules this list obeys.** The exact DDA, the batched deposit, the
Jeffreys priors, the lock-witness pattern and the walk margin are settled
mechanisms; nothing below touches them. Refactors are proven byte-identical,
not argued. Code comments carry no document paths, no measurement numbers and
no negative results; those go to memory and `REVIEW_LOG.md`.

---

## Index

| id | tier | item | status |
|---|---|---|---|
| A1 | A | `robot_id` parameter is never read | open |
| A2 | A | `topk_probs_max_k` parameter is never read | open |
| A3 | A | three `scovox::Params` fields are declared, warned about and inert | open |
| A4 | A | vestigial `std::pair` returns on two publishers | open |
| B1 | B | the `dp` declare-parameter lambda is copied four times | open |
| B2 | B | `frameId()` exists and is bypassed four times | open |
| B3 | B | `SP` and `ISP` are the same alias | open |
| B4 | B | four hand-rolled TF lookups; one fallback fails silently | open |
| B5 | B | the two LiDAR point loops share a copied prefix | open |
| B6 | B | the 14- and 16-field PointCloud2 schemas are written out twice | open |
| B7 | B | the TSDF and fine-TSDF cloud publishers are one body | open |
| B8 | B | one-line duplicate stamp-gap computation | open |
| C1 | C | `publishPlanningMap` waits on TF for 50 ms under the write lock | open |
| C2 | C | `publishScovoxMap` has no subscriber-rise re-arm | open |
| C3 | C | `publishFineTSDFPointCloud` has no dirty gate at all | open |
| C4 | C | `publish_pointcloud:=false` still advertises the topic | open |
| C5 | C | `have_di_` is an atomic guarding a non-atomic copy | open |
| C6 | C | `carve_band` was applied by moving the ray origin: TSDF sign wrong in the strip before the hit, carve set clipped at the window's far edge | **closed 2026-09-07** — a `carve_reach` argument on the true origin; E-W34 |
| D1 | D | one class does eleven jobs; four methods are 280–430 lines | open |
| D2 | D | extract the gyro deskew into its own class | open |
| D3 | D | extract the frame-admission gate | open |
| D4 | D | extract the wire chunk interleaver as a testable free function | open |
| D5 | D | make the planning-map projection a free function over the grid | open |
| D6 | D | the snapshot seam is ceremony: pack 14 scalars, unpack 11 aliases | open |
| D7 | D | mixed member initialisation and multi-declarator lines | open |
| D8 | D | dense one-liners and an un-indented else branch | open |
| E1 | E | 36 history-tag comment lines | open |
| E2 | E | 13 comment lines cite scripts or config files by path | open |
| E3 | E | measurement numbers in comments | open |
| E4 | E | stale comments that claim dead fields are read | open |
| F1 | F | `range_decay_length` is named as a weight, acts as a cull switch, and its header says "caller-applied" with no caller | open — pairs with M3 |
| F2 | F | the replay still writes `0xFFFF` where the core has a named sentinel | open |
| F3 | F | `scovox_cleanup.md` carries a correction that the tree then reversed, and nothing points here | open |
| F4 | F | Beta `uint16` ceiling on an unbounded node run | closed — halving at 0.9·kMax is unconditional |
| F5 | F | `w_occ:w_free` at 1:1 | deferred experiment; storage motive gone |
| F6 | F | Dir grid to `uint16` | closed — `s_total` unbounded while the band is un-batched (M9) |
| F7 | F | "comment the dead sites" (cleanup items 3, 4, 6) | withdrawn under the standing rule |
| G1 | G | one source of parameter defaults (M1) | open; decide the canonical struct |
| G2 | G | six library knobs as ROS parameters (M2) | open |
| G3 | G | dscovox fused Dir grid at `dir_leaf_bits`; first-pose cache (M5) | open; decide |
| G4 | G | block-run comments; the M6 hazard is withdrawn | open; comment only |
| G5 | G | wire quantisation scale vs evidence cap (M7) | open |
| G6 | G | `semantic_confidence` on the posterior marginal (M11) | open; decide, the wire value changes |
| G7 | G | nine stale byte-size / type-name comments (L1) | open |
| G8 | G | `.md` references in `scovox_slot_rules/scripts` (L2) | open; other repo |
| G9 | G | legacy substrate: keep as ablation or delete (L4) | open; decide |
| G10 | G | `wire_study.py` still documents v5 (L5) | open; doc-only |
| G11 | G | `downsample_voxel_size` default said two ways (L6) | open |
| G12 | G | `FarCarveBitIdenticalToFullWalk` red at band 0.30 (M4) | open; decide |
| G13 | G | four non-compiling commits in the S-series (errata) | decide before the next push |
| G14 | G | sparse-branch counters compiled out | landed (`a062908`); proof items (1), (2), (5) open |
| G15 | G | `buildSwitches()` names seven of the nine live switches | **closed 2026-09-07** — all ten are on the line |
| G16 | G | a `WALKER_TIMERS=0` build silently zeroes `tsdf_ms` / `sembeta_ms` | open; decide whether the node should refuse to publish them |

---

## A — Dead code (behaviour-neutral)

- [ ] **A1 — Delete the `robot_id` parameter and member.**
  Declared at `:490` (`robot_id_ = dp("robot_id", std::string(""))`), stored
  at `:3115`, read nowhere. Before deleting, grep `config/` and `launch/` for
  `robot_id`; a yaml that still sets it is silently ignored by rclcpp, so this
  is a cleanliness question for the configs, not a breakage.

- [ ] **A2 — Delete the `topk_probs_max_k` parameter and `topk_topk_max_`.**
  Declared at `:830`, member at `:3182` (`int topk_topk_max_{5}`), never read.
  `TopkProvider` (`topk_provider.hpp:36`) takes `(logger, clock, dir, max_sem)`
  and exposes only `setTopkTrunc(int)` at `:49`; there is no place this value
  could go. If a max-K clamp was intended, it belongs in `TopkProvider`, not
  as a dead node member.

- [ ] **A3 — Stop declaring the three `scovox::Params` fields that nothing
  this node links reads.**
  `semantic_top_k` → `P.top_k` at `:372-379` (a clamp against `K_TOP` and an
  `RCLCPP_WARN` on a value nobody consumes); `grazing_angle_threshold` at
  `:417`; `band_only_integration` at `:395`, which is only echoed in the
  startup log at `:299`. The sole reader of `band_only_integration` is the
  legacy `scovoxmap.cpp:271`, which `scovox_mapping_node` does not link
  (`CMakeLists.txt:40` builds it as a separate library; `:60` links the node
  without it). Nothing reads `top_k` or `grazing_angle_threshold` anywhere.
  *Do:* remove the three `dp()` calls and the clamp/warn block; drop the log
  field at `:299`. Removing the fields from `scovox::Params` itself is L4 of
  the 2026-09-04 review (legacy map types) and should go with it, not here.
  E4 fixes the comments that claim otherwise. *Since the addendum above:* if
  this parameter is kept rather than deleted, it must be renamed or re-pointed
  — `semantic_top_k` is now the obvious name for the ingest cap that ships, and
  leaving it wired to a field nothing reads is a worse smell than it was.

- [ ] **A4 — Drop the vestigial `std::pair<size_t,double>` returns.**
  `publishScovoxMap` (`:2088`) returns a pair whose double is hard-coded
  `0.0`; the timer at `:213` discards the whole value. `publishBinaryMap`
  (`:2209`) returns `{emitted, bytes_MB}`; the scan tail keeps the first and
  casts the second to void at `:1301`, and the timer at `:229` discards both.
  *Do:* `publishScovoxMap` → `void` or `size_t`; for `publishBinaryMap`
  either log the MB figure where it is computed or return `size_t` only.

## B — Redundancy (behaviour-neutral)

- [ ] **B1 — One declare-parameter helper instead of four lambdas.**
  The identical lambda `auto dp = [&](auto n, auto d){ return
  this->declare_parameter<decltype(d)>(n, d); }` appears at `:359`, `:459`,
  `:854` and `:991`. *Do:* a private member template
  `template<class T> T dp(const std::string& n, T d)`; the four bodies then
  call it unchanged.

- [ ] **B2 — Use `frameId()` at the four sites that inline it.**
  The helper at `:1038-1039` masks `header.stamp.nanosec & 0xFFFF`; the same
  expression is written out at `:1234` and `:1339` (depth) and `:1610` and
  `:1830` (cloud). Byte-identical result; the helper simply becomes the one
  place the frame-id convention lives.

- [ ] **B3 — Delete the local `SP` alias.**
  `:984` declares `using SP = ApproximateTime<Image, Image>` inside
  `setupSubscribers`; the member alias `ISP` at `:3371` is the same type and
  is what the synchroniser member is declared with. Use `ISP`.

- [ ] **B4 — One TF-lookup-with-fallback helper, and no silent failure.**
  `onPointCloud` writes the "exact stamp, then `Time(0)` with 50 ms" policy
  out twice: `:1854-1864` for `T_oi` and `:1868-1874` for the observer
  frame. `onImages` has its own two lookups at `:1368` and `:1378`. About 40
  lines collapse into one `std::optional<Eigen::Isometry3f>
  lookupOrFallback(target, source, stamp, timeout)`.
  The second block in `onPointCloud` ends `} catch (...) { return; } }` at
  `:1874`: a failed observer-frame lookup drops the scan with no warning and
  no `tf_fallback_count_` increment, while the first block warns and counts.
  Give both the same diagnostics. This is the one item in tier B with a
  user-visible effect (a new throttled warning), and it is the intended one.

- [ ] **B5 — Share the per-point prefix between the two LiDAR loops.**
  `integrateLidarSnapshot` has a downsample branch and a per-point branch
  (`} else {` at `:1774`) that both do: read xyz, `isfinite`, squared-range
  gate against `rr2`, deskew via `deskewRot`. Only the tail differs
  (accumulate into the medoid grid vs integrate directly). *Do:* one lambda
  `auto parsed = pointAt(i)` returning `std::optional<Eigen::Vector3f>`,
  called from both branches.

- [ ] **B6 — Build the PointCloud2 field list once.**
  `publishPointCloud` calls `setPointCloud2Fields(16, …)` at `:2814` and
  `setPointCloud2Fields(14, …)` at `:2829` with the first 14 fields repeated
  verbatim. The comment at `:2803` already warns that the two lists and the
  `sem_cnt*/sem_cls*` writers must be kept in step, which is the smell
  naming itself. *Do:* a `std::vector<PointFieldSpec>` of the 14 common
  fields, push the two optional ones, apply once. Keep the `K_TOP`
  `static_assert`s; they are the right guard.

- [ ] **B7 — Merge the TSDF and fine-TSDF cloud publishers.**
  `publishTSDFPointCloud` (`:2966`) and `publishFineTSDFPointCloud` (`:3052`)
  differ only in grid, resolution, publisher and gate. *Do:* one
  `publishZeroCrossingCloud(grid, res, pub, gate)`; C3 then gets its gate for
  free.

- [ ] **B8 — Compute the seg/depth stamp gap once.**
  `:1347` evaluates the same stamp difference twice on one line to build one
  log message. Bind it to a local.

## C — Inconsistencies that change behaviour when fixed (decide, then do)

- [ ] **C1 — `publishPlanningMap` waits on TF for up to 50 ms while the scan
  tail holds the map write lock.**
  `finishScanTail` calls it at `:1315` from inside the `MapWriteLock`; in
  rolling mode it runs `lookupTransform(int_frame_, base_frame_, Time(0),
  Duration::from_seconds(0.05))` at `:2644-2646` every scan. `publishBinaryMap`
  handles the identical situation with `rclcpp::Duration(0, 0)` and says why
  at `:2270-2272`: a timeout only matters while the buffer is still empty at
  startup, and burns the wait under `map_mtx_`.
  *Change:* use a zero timeout, warn-throttled skip on miss, exactly as
  `:2273-2280` does. *Effect:* a startup tick may skip one planning frame it
  previously waited for; steady state is unchanged. Also give the method a
  `const scovox::MapLockHeld&` parameter like the other three publishers; it
  reads `split_map_` and currently takes no witness.

- [ ] **C2 — Give `publishScovoxMap` the subscriber-rise re-arm the cloud
  publishers already have.**
  `:2089-2090` is `subscription_count()==0 → return; !sm_dirty_.exchange(false)
  → return`. `publishPointCloud` (`:2780-2788`) and `publishTSDFPointCloud`
  (`:2973`) track `*_prev_subs_` so a consumer that attaches after the map
  goes quiescent still receives one frame. The `ScovoxMap` topic does not,
  and its QoS is not transient-local either. *Effect:* one extra publish per
  subscriber rise. Fold this into a small `PublishGate { std::atomic<bool>
  dirty; size_t prev_subs; bool pass(const PublisherT&); }` and use it in all
  four publishers, which also removes three copies of the gate logic.

- [ ] **C3 — Give `publishFineTSDFPointCloud` a dirty gate.**
  `:3052-3054` gates on publisher existence and subscriber count only, then
  re-walks the whole fine grid every viz tick. `markMapDirty()` at
  `:2079-2083` sets `sm_dirty_`, `pc_dirty_` and `tsdf_pc_dirty_` and nothing
  for the fine cloud. *Effect:* fewer publishes on an unchanged map; no
  change to any published frame's content. Falls out of B7 + C2's gate type.

- [ ] **C4 — Create `pc_pub_` only when `publish_pointcloud` is true.**
  `:1022` creates it unconditionally; the timer tests `pub_pc_` at `:214` and
  the function re-tests `!pc_pub_` at `:2773`. `pl_pub_` (`:1024`) and
  `tsdf_pub_` (`:1026`) are created conditionally. *Effect:* with the flag
  off the topic is no longer advertised, which is what the flag says. The
  timer test becomes `if (pc_pub_)` and `pub_pc_` can go.

- [ ] **C5 — Decide what protects `di_`.**
  `have_di_` is `std::atomic<bool>` with acquire/release (`:959`, `:1340`,
  `:3315`) but `di_` is a plain `CameraInfo` copy written at `:959` and read
  in `onImages`. Both callbacks sit in the default mutually-exclusive group,
  so they never run concurrently and the atomic is neither needed nor, on its
  own, sufficient. *Options:* (a) plain `bool` plus a one-line comment
  stating the group guarantee; (b) a mutex around both if either callback is
  ever moved to `viz_cb_group_` or a reentrant group. Recommend (a); it
  states the actual invariant.

- [x] **C6 — `carve_band` was applied by moving the ray origin; it is now a
  carve reach measured from the hit, on the true origin.** *Closed
  2026-09-07.* The node's `integrateHit` used to build
  `co = O + (Hp − O)·(rng − carve_band)/rng` and pass `co` as the ray origin,
  so the walker signed every voxel's TSDF against a point `carve_band` before
  the hit. Two consequences, both confined to the KITTI launch
  (`semantickitti_eval.launch.py` defaults `carve_band` to 0.1; every other
  yaml and launch file is −1.0 = full ray, where the mechanism is inert): the
  strip between `carve_band` and `sdf_trunc` in front of the hit was written
  with a negative distance, and off-axis voxels near the window's far edge
  read as behind the surface and were neither carved nor written.
  *Change:* `ScovoxMapSplit::integrateHit` takes a trailing `carve_reach`
  (default 0 = full ray). The fused walker bounds the carve gate and the walk
  start with it and keeps `origin` as the sensor; `far_carve` stands down when
  a window is set, since its reduction assumes the walk starts at the sensor.
  The split walker starts `carveRay` at the window's near edge and keeps the
  TSDF origin. The node passes `carve_band_` as the reach and no longer
  computes a range for it; the RGB-D cull keeps its own. `replay_kitti` gained
  `--carve-band`, `--carve-band-origin-trick` (the former mechanism, kept only
  so the two can be diffed) and `--dump-tsdf`.
  *Proof:* full-ray dumps byte-identical to pristine HEAD, readout and slots
  (KITTI 08, 100 frames, res 0.05, TSDF on). At `carve_band` 0.1 the Beta
  carve set and the TSDF voxel set are supersets of the old result, no stored
  TSDF distance falls beyond a rounding tail, and the Dir grid follows Beta
  through the `dirichlet_min_p_occ` deposit gate; the counts are in
  `scovox_slot_rules/REVIEW_LOG.md` E-W34. Tests:
  `ScovoxMapSplitCarveReach.*` in `test_scovox_map_split.cpp` assert the
  containments, the full-ray inertness and the far-carve stand-down.
  *Consequence, not yet decided:* the old mechanism under-carved by accident,
  and the window the parameter names costs union mIoU on KITTI 08 at 10 cm
  (E-W34), on a scorer that cannot reward a carve. Whether the launch keeps
  `carve_band` 0.1 is a configuration decision, open.

## D — Structure (moves code; prove byte-identical)

- [ ] **D1 — Acknowledge the shape before cutting.**
  One class holds parameter handling, RGB-D and LiDAR pipelines, gyro
  deskew, medoid downsample, four viz publishers, the wire path with
  chunking, byte budget and heartbeat, the planning-map projection, mesh
  extraction and memory logging. Measured method lengths:

  | method | lines | at |
  |---|---|---|
  | `publishBinaryMap` | 428 | `:2209-2637` |
  | `declareNodeParams` | 395 | `:458-853` |
  | `integrateLidarSnapshot` | ~288 | `:1538-1825` |
  | constructor | ~279 | `:67-345` |
  | `publishPointCloud` | 194 | `:2772-2965` |
  | `publishPlanningMap` | 135 | `:2637-2771` |

  D2–D5 are the four cuts that come out cleanly; each is independent and
  should land as its own commit with its own byte-identity check.

- [ ] **D2 — Extract `LidarDeskewer`.**
  `onImu`, `ensureLidarImuExtrinsic`, `buildDeskewTable`, `quatExp`,
  `decodePointTimeOffset`, the `deskewRot` lambda and roughly twelve members
  (IMU ring buffer and its mutex, extrinsic cache, table, window, gyro bias,
  translation-deskew flag). The class owns the IMU buffer and exposes
  `bool prepare(t0, window)` and `Eigen::Matrix3f rotAt(dt)`. The node keeps
  the subscription and the TF lookup for the extrinsic.

- [ ] **D3 — Extract the frame-admission gate.**
  `tfGatePass` (startup stabilise + runtime jump re-arm), `locRejecting`
  (`/alignment_status` DiagnosticArray parse, `:1155-1157`), `admitFrame`
  (`:1180`), `onAlignmentStatus`, and the `loc_*`, `gate_*` members. Pure
  state machine over `(origin, now)`; unit-testable without a node.

- [ ] **D4 — Extract the wire chunk interleaver from `publishBinaryMap`.**
  About 90 lines of proportional-interleave chunking, the per-tick byte
  budget and the FIFO `share_deferred_` queue. Make it a free function in
  `node_utils.hpp` taking `(blocks, budget_bytes, deferred&)` and returning
  the emit list; add a case to `test/test_publish_gate.cpp` (or a sibling)
  so the budget and FIFO order are tested, which today they are not.

- [ ] **D5 — Make the planning-map projection a free function over the
  Beta grid.**
  `publishPlanningMap` mixes the crop decision (fixed vs rolling, the TF
  lookup from C1), the terrain-relative z-band, the occupancy projection and
  the inflation loop (`:2757-2761`, four dense lines). Split into
  `projectPlanningGrid(grid, box, params) → OccupancyGrid` (free, testable)
  and a thin node method that resolves the box and publishes. `mode_ ==
  "rolling"` is a string compare per call; cache a `bool rolling_` at
  parameter time.

- [ ] **D6 — Simplify the snapshot seam.**
  `DepthSnapshot` (`:1191`) is filled with 14 scalars at `:1387-1392` and
  immediately unpacked into 11 `const` aliases at `:1213-1222` in its single
  consumer; `LidarSnapshot` (`:1519`) is the same pattern, and `onPointCloud`
  re-aliases `res.do_deskew`, `res.ds_in`, `res.ds_out` for one log line.
  The rule the seam encodes ("no TF inside integrate") is a discipline the
  struct does not enforce. *Do:* drop the alias block and read `s.field`
  directly, or give the struct the per-frame constants as a nested
  `FrameConsts` so the integrate signature is `(const FrameConsts&, const
  Image&, …)`. Lowest priority in the tier; the cost is reading, not running.

- [ ] **D7 — One member per declaration, every member initialised.**
  `:3115` declares nine strings on one line; `:3181-3184` mixes
  default-initialised and uninitialised members of three types
  (`int max_sem_, stride_{1}; … bool trace_nr_{false}, pub_pc_,
  pub_plan_{false}, pub_tsdf_{true};`). Give every member a default and its
  own line. Behaviour-neutral because every one of them is assigned in the
  constructor before use, but a future reorder would not be.

- [ ] **D8 — Reformat the dense lines, in one commit by themselves.**
  42 non-comment lines carry three or more statements; 7 exceed 160
  columns; the `else` at `:1774` opens a ~50-line body that is not indented
  relative to its `if`; the inflation loop at `:2757-2761` packs four nested
  `for`s into four lines. A formatting-only commit keeps the noise out of
  the diffs for A–D.

## E — Comment hygiene (standing rules)

- [ ] **E1 — Rewrite the 36 history-tag comment lines to say what the code
  does now.**
  Patterns present: "audit item N" (several, e.g. `:2784`), "Run 4 review"
  (`:2774`), "code-smell fix 2026-08-26", "Step 12.10 (2026-05-09)", "(was
  10)", "was two full forEachCell walks", "the old post-transform gate", "the
  legacy code froze", "pre-refactor". Each describes a diff, which git
  already holds, or a negative result, which belongs in `REVIEW_LOG.md`.
  Keep the *reason* (e.g. "KeepLast(1) reliable, not transient_local, so a
  late subscriber would wait forever") and drop the provenance tag.

- [ ] **E2 — Decide whether the no-doc-pointers rule extends to scripts and
  configs, then apply it.**
  13 comment lines cite `tools/tsdf_parity_test.py`,
  `eval_e13_byte_parity.py`, `pointcloud_to_npz.py`,
  `config/scovox_best_method.yaml`, `config/scovox_rgbd_complete.yaml`,
  `config/scovox_rgbd_unbatched.yaml`, `scovox_lidar_raw_deskew.yaml`,
  `config/exploration_fused_bag.yaml`. The 2026-09-04 sweep (L2) removed
  `.md` paths only. Script names are the same kind of pointer (they move, and
  two of these already live outside this repo); config names arguably
  document which deployment exercises a branch. Recommendation: remove the
  script names, keep config names only where the comment is "this branch is
  reached by config X".

- [ ] **E3 — Remove measurement numbers from comments.**
  `:62` ("measured ~44 s / 176 frames"), `:967` ("176 frames
  (scenenet_soft_e2/0_182), 161 (s1_ktop/K20)"), `:1362-1364` ("~250 ms …
  ~5% of frames in testing"), `:1922` ("acceptance gate 15%"), `:2008`
  ("matching the production mIoU baselines"). These are results; the rule
  puts them in memory and `REVIEW_LOG.md`. Where the number drives a
  default (the 15 % gate), the default itself is the record and the comment
  should say what the knob does, not what was measured.

- [ ] **E4 — Fix the three comments that claim dead fields are read.**
  `:72` ("… grazing_angle_threshold, semantic_occ_gate, resolution, top_k)
  that the …"), `:113`, and `:3231-3232` all state that these `Params`
  fields are consumed by the integration and publish paths. After A3 they
  are not declared; before A3 they were never read. Do with A3.

---

## F — Items carried over from `scovox_cleanup.md` (re-verified 2026-09-06)

**What that document is.** A 2026-09-02 work list of simplifications in the
shipped deposit path, corrected the same day after a repo-wide caller check
("no code deletion survived that check"), and annotated once on 2026-09-04.
It tags each item DEAD / INERT-IN-CONFIG / LIVE ELSEWHERE. Its central
lesson, that "dead in the SceneNN suite" and "dead in `scovox`" are different
claims, stands and is why every item below names the scope it was checked in.

**What changed under it.** The corrections of 2026-09-02 argued *against*
deleting the `quality` threading and the `range_decay_length` field because
the ROS node used them for range decay. Two days later `1101e53` deleted the
`quality` parameter from `integrateHit`/`integrateMiss`, `semHit`, `semBand`
and `semCarve` anyway, and the node's `exp(-r/L)` computations went with it.
The field survived; the feature did not. Nothing in `scovox_cleanup.md`
records this, so a reader today is told to protect a mechanism that no longer
exists. That is F1 and F3.

### Status of every item

| cleanup item | claim (as written) | status at `77b99da` | evidence |
|---|---|---|---|
| 1 `quality` | inert here, live range-decay channel in the node; do NOT delete | **overtaken: deleted by `1101e53`** | `scovox_map_split.hpp:161-165` and `:747-759` take no `quality`; the node calls `integrateHit(O, Hp, rng, cp, prof)` at `:1272`, `:1779`, `:1823`; `grep 'std::exp(-'` on the node is empty |
| 2 `kappa0` | 1.0 everywhere; `class_share = kappa0·p_occ_post·quality` collapses to `p_occ_post`; say so, keep the flag | **stands, now structural** | with `quality` gone the deposit is `kappa0·p_occ_post` by construction; `sem_split_map.hpp:188`, `:196` (default 1.0), `:203` already say so; replay default 1.0 at `replay_scenenn.cpp:156` |
| 3 `prof` always null | DEAD in the replay, LIVE for fusion; comment each site | **stands as a fact; the action is withdrawn (F7)** | 12 `prof ?` sites now (`sem_split_map.cpp:421-877`), replay passes none; the node passes `rgbdProf()`/`lidarProf()`, so `prof` is non-null in the deployed mapper |
| 4 three dead branches | BKI kernel, `semantic_spread_radius`, NAIVE/MAJORITY_VOTE; flag, don't delete | **stands; action withdrawn (F7)** | `sem_split_map.cpp:612`, `:731`, `:707`/`:716`; the two modes are part of L4 (2026-09-04 review) |
| 5 `range_decay_length` | caller-applied via `quality`; do NOT delete the field | **half-overtaken: field kept, weight gone** | core reads it only to clamp (`sem_split_map.cpp:269`); header still says "caller-applied" (`sem_split_map.hpp:406`); node uses it as the range-cull switch at `:1271`, `:1578` and says so at `:1269-1270` |
| 6 wall guard | DEAD under `batch_free_carve`; comment it | **done** | `sem_split_map.cpp:463` returns first; `:468-470` states the guard is off by default and for direct callers |
| 7 `nhit` | pure write traffic in the candidate; add a `TRACK_NHIT=0` deployment build | **done, further than proposed** | `SCOVOX_TRACK_NHIT` defaults 0 (`dir_voxel.hpp:121-122`); `build_rules.sh:99` `TRACKS` carries only `QMAX`; DirVoxel 20 B (`:232`, `:241`); `slot_readout.py:71-73` records the `has_nhit` reader fix |
| 8 Beta integers in floats | 8 B → 4 B needs hit batching first; must be measured | **done, and measured** | `batch_hits = true` (`sem_split_map.hpp:405`), `SCOVOX_BETA_U16 = 1` at scale 8 (`beta_voxel.hpp:70`, `:80`); the "semantic change that must be measured" is H4 of the 2026-09-04 review: batching alone is 91–96 % of the drift from the published numbers |
| 9 DirVoxel | `cnt` overflows u16 3.2x; batching bounds it; re-derive against `s_total` | **closed (F6)** | `s_total` is the stored word (`dir_voxel.hpp:158`); the band is un-batched (M9), so `s_total` has no per-frame bound |
| 9a `cls` as u8 | REJECTED: 0.67 MB against 31 silent-failure sentinel sites | **rejection stands; hazard shrank** | `0xFFFF` literals: `dir_voxel.hpp` 0 (was 15), `sem_split_map.cpp` 0 (was 11), `replay_scenenn.cpp` 5 (unchanged); the core names the sentinel `kEmptySlot` (`voxel.hpp:55`) |
| 9b `nhit` saturates | affects `VICTIM_MEAN` / `ADMIT_NORM` research arms | **moot** | `nhit` is compiled out by default; the two flags are referenced only from `dir_voxel.hpp` and `version.hpp` |
| order 1–3 comments | comment null-`prof`, dead modes, wall guard | **withdrawn (F7)**; wall guard already commented | standing rule: no negative results in comments |
| order 4 build split | `TRACK_NHIT=0` deployment build | **done** (item 7) | |
| order 5 batch + u16 | needs an experiment | **done** (item 8), cost recorded in H4 | |
| order 6 `w_occ:w_free` 1:1 | needs an experiment | **deferred (F5)** | `REVIEW_LOG.md` has no 1:1 arm |
| order 7–9 withdrawn | keep `range_decay_length`, keep `quality`, keep `cls` u16 | 7 and 8 were **reversed by `1101e53`** for `quality`; 9 stands | |

### Open

- [ ] **F1 — Rename or restore `range_decay_length`; fix the header that
  claims a caller applies it.** Pairs with M3 of the 2026-09-04 review and
  should be decided with it. Today the parameter is a boolean in disguise: the
  node reads it at `:1271` only as `> 0` to enable the
  `min_range`/`max_range` cull, and copies it into `SP.semsplit` at `:128`
  where the core clamps it (`sem_split_map.cpp:269`) and reads it nowhere
  else. `sem_split_map.hpp:406` still documents `exp(-r/L)` "caller-applied";
  no caller applies it since `1101e53`. `split_memory_demo.cpp:97` sets it to
  50 inertly. Eight launch/config files set it (`scenenn_eval`,
  `scenenet_eval`, `scenenet_eval_fusion`, `semantickitti_eval`,
  `scovox_single_robot` launch files; `lidar_mapping`, `scovox_bin_min`,
  `default_params` yamls) and two mapping tests reference it. *Decide:* (a)
  rename to `enable_range_cull` (bool), delete the `SemSplitParams` copy and
  the header line, update the eight files and two tests in the same commit;
  or (b) restore the weight at the node's two hit sites. (a) matches the
  measured record (E7: the decay arm was a null). Either is a config-surface
  change, so it belongs in tier C's decision list, not tier A.

- [ ] **F2 — Use `kEmptySlot` in the replay instead of `0xFFFF`.**
  `replay_scenenn.cpp:803` (init), `:822` (filled-slot test) and `:849`
  (record sentinel) still spell the literal; `:750` is a comment. `:989`
  is an unrelated `0xFFFFFFFF` trace-counter clamp. The core is already
  clean. Behaviour-neutral; a `.slots` md5 on one scene proves it. Lives in
  `scovox_slot_rules`, not this repo.

- [ ] **F3 — Put a status banner on `scovox_cleanup.md`.** Its 2026-09-02
  corrections ("do NOT delete the internal threading", "do NOT delete the
  field") describe a tree that `1101e53` changed two days later, and the
  document has no pointer to this section or to M3. Three lines at the top,
  the way `docs/archive/code/code_review_2026_09_04.md` handles its predecessor.
  Doc edit in `scovox_slot_rules`; not done here because it is another
  repo's ledger.

### Closed on evidence

- **F4 — Beta `uint16` ceiling on an unbounded node run: handled.** The
  cleanup document bounds `a_occ` by `prior + w_occ·frames` for a 1 300-frame
  replay. The node runs indefinitely, and `evidence_saturation` is 0 in all
  three RGB-D yamls, so the question was whether a fixated voxel can hit the
  fixed-point clamp (`pack()` clamps at 65 535 counts, `beta_voxel.hpp:145`;
  8 191.9 real units at scale 8, about 4.2x over the replay's 1 951).
  It cannot: `applyBetaSaturation` (`sem_split_map.cpp:1104-1116`) halves
  both parameters at 0.9·kMax **unconditionally**, before and independent of
  the opt-in `evidence_saturation` cap, and the halving preserves `p_occ`.
  Recorded so the ceiling is not re-raised as a node defect.

- **F5 — `w_occ:w_free` at 1:1: the storage argument is gone.** The 1/8
  lattice holds 1.5 exactly (`beta_voxel.hpp:183`), so no bit is recovered
  by going to 1:1. What remains is a model question the ring left "not
  demonstrated" at +0.0102 mIoU 6/2. `REVIEW_LOG.md` has no 1:1 arm. It is a
  deferred experiment, run on request; it is not a cleanup item.

  **E-W32 (2026-09-07) narrows the free-space half of this.** An 8-scene
  paired sweep prices `w_free = 0` outright: 3.464x faster (8/8) for
  **−0.0385 union mIoU against tuned SLIM-VDB, losing 0/8**, and −0.0257
  against shipped scovox. So the free-space term is doing real semantic work,
  and any 1:1 experiment must be scored on union mIoU across all eight scenes,
  never on scene 016 — 016 is the one scene whose sign disagrees with the set.
  Turning the carve off is **not** a shipping default on this evidence.

  **E-W33 (2026-09-07) corrects how that comparison is described.** The scovox
  arm in those sweeps runs on driver defaults — `sem_band` 0.10, `w_occ` 1.5,
  `evict_by_confidence`, `dump_gate` 0.5 are all defaults in
  `replay_scenenn.cpp`, not overrides — so the comparison is not "tuned scovox
  against untuned SLIM-VDB". The asymmetry is that SLIM-VDB's published
  `min_weight` 20 is a raw hit count and does not transfer to our sequence
  lengths. Both the defaults-against-published (+0.0946, 7/8) and
  tuned-against-tuned (−0.0127, 3/8) comparisons are reportable; neither alone
  is the result.

- **F6 — Dir grid to `uint16`: closed.** The 2026-09-04 annotation in the
  cleanup document already asked for the bound to be re-derived against
  `s_total`. It cannot be bounded per frame: the band deposit is un-batched
  (M9 of the 2026-09-04 review, deferred), so `s_total` grows with rays, not
  frames, and the measured look counts reach 332 622. The payoff was 2.68 MB
  across eight scenes, 1.8 % of voxels. Not worth a storage mode that M9 would
  have to invalidate.

- **F7 — "Comment the dead sites" (cleanup items 3, 4, 6; order rows 1–3):
  withdrawn.** Reachability in one driver is a property of that driver, and
  in the deployed node `prof` is non-null. A comment saying "dead in the
  replay" is a negative result about a caller, which the standing rule keeps
  out of code. The one site where a comment earns its place, the wall guard,
  already has it (`sem_split_map.cpp:468-470`).

---

## G — Findings carried over from `docs/archive/code/code_review_2026_09_04.md` (re-verified 2026-09-06)

**What that document is.** A tree-wide review written on 2026-09-04 and
extended through 2026-09-06: four High, eleven Medium and seven Low findings
(H1–H4, M1–M11, L1–L7), a code-smell sweep of the full tree (S1–S11), an
errata section on its own claims, and six addenda that are experiment
write-ups rather than code findings. It carries a status line per item. This
tier does not restate it; it re-reads every status against `77b99da`, records
where the tree and the status disagree, and turns what is still open into
checkboxes with the proof each one needs.

**What changed under it.** The S-series was fixed in the commits
`f202860..32121f2` on 2026-09-05 and every fix is still in the tree. The
H/M/L items marked open are, with two exceptions, still open and unchanged:
M6's stated impact does not survive a re-read (G4), and the fold-order half of
M5 is now documented (G3). Nothing else from the H/M/L series has moved since
the review was written. The edit that became `a062908` described in the
second addendum is the review's own closing recommendation being acted on
(G14); it is recorded here, not started here.

### Status of every item

| id | finding (as written) | status as written | status at `77b99da` | evidence |
|---|---|---|---|---|
| H1 | node defaults are not the promoted configuration | resolved (`33aa576`) | resolved | `config/scovox_best_method.yaml` present; the other two RGB-D yamls cite it |
| H2 | the promoted state is uncommitted | resolved (`1101e53`, `33aa576`) | resolved | both commits in `git log` |
| H3 | `integrateHitSplit` ignores `tsdf_enabled` | resolved (`f2d535e`) | resolved | `scovox_map_split.hpp:289` and `:564` test `tsdf_enabled_` |
| H4 | published numbers do not describe the tree | open, needs a re-run | **open; an experiment, not a code item** | tracked in memory (`published-numbers-predate-exact-dda`); the review's own addendum reproduces the `total` column from `cells/goal8.tsv`. Not on this list |
| M1 | four parameter structs, four sets of defaults | open | **open, unchanged** | no `promotedParams` symbol anywhere; `struct Params` at `map_interface.hpp:28`, `sem_split_map.hpp:149`, `scovox_map_split.hpp:40`, `tsdf_map.hpp:48`; replay `Args` at `replay_scenenn.cpp:42`; node `dp()` defaults at `:362-369` |
| M2 | six library knobs have no ROS parameter | open | **open, unchanged** | no `dp("…")` for any of the six; the node prints two of them (`:298`, `:342`) and declares neither; no eval launch file takes `semantic_band_length`; `w_occ` defaults 6.0 in two launch files, `kappa0` 2.0 in one |
| M3 | `range_decay_length` dead in the split path | open | open → **F1** | tier F |
| M4 | `FarCarveBitIdenticalToFullWalk` red at band 0.30 | open | **open; not re-run (no builds this session)** | `test_scovox_map_split.cpp:887`; memory `farcarve-identity-fails-at-head`; suite baseline 183/184 |
| M5 | dscovox fused grids ignore `dir_leaf_bits`; first pose wins; fold order matters | open | **two of three open** | both fused grids built with `P.leaf_bits` (`dscovox_node.cpp:415-443`); the first-pose cache is still a banner-documented c-slam hazard (`:26-34`, `:98-102`); the fold-order dependence is now documented at `consensus_merge.hpp:141-150` |
| M6 | wire block runs hard-code `leaf_bits = 3` | open | **downgraded to a comment fix (G4)** | encoder `binary_serializer.hpp:567-569` and decoder `:697-700` tile coordinate space (`& 7`, `bx*8+lx`); `leaf_bits` enters neither; only `:20` and `:530` call the tile a leaf block |
| M7 | `evidence_saturation` is one knob for two caps | partly resolved (documented) | **unchanged** | `dp("evidence_saturation", 1000)` at `:404`; `class_evidence_saturation = -1` follows the shared cap (`sem_split_map.hpp:239`, `sem_split_map.cpp:1130-1131`); no `wire_evidence_scale` in the tree |
| M8 | write-ups print the demoted build flag | resolved | resolved as written | a results-doc item; not re-verified |
| M9 | `batch_hits` stages the endpoint only; band un-batched | open, deferred | deferred → **F6** | memory `band-is-unbatched`; the three candidate arms run on request only |
| M10 | semantic-uncertainty basis | resolved (E12) | resolved as written | memory `semantic-uncertainty-is-degenerate` |
| M11 | published `semantic_confidence` uses the rejected basis | open | **open, unchanged** | `node_utils.hpp:326-329`: `(best_cnt + 1) / (sum_cnt + n_active + effectiveResidual + 1)`; consumers `scovox_node.cpp:2925`, `dscovox_node.cpp:706` |
| L1 | stale byte-size / type-name comments | open | **open; all nine rows still stale, one moved** | `sem_split_map.cpp:690` is now `:705`; `SemBetaMap` and `SemDirMap` are declared nowhere in the tree |
| L2 | in-code references to moved or missing documents | partly resolved (code swept) | **scripts half unchanged** | `scovox_slot_rules/scripts`: `DESIGN.md` ×12 (never existed), `FINDINGS.md` ×9 and `PLAN.md` ×7 (both in `archive/`), across 17 files; `a.md` is an attribute, not a file |
| L3 | README document index | resolved | resolved as written | not re-verified |
| L4 | legacy voxel / map types compiled and tested | open | **open, unchanged** | `voxel.hpp`, `sembeta_voxel.hpp`, `scovoxmap.hpp/.cpp` present; the node routes `naive` / `majority_vote` at `:419-420`; `test_beta_update` 32 cases (31 + one from S11), `test_consensus` 25 |
| L5 | `wire_study.py` documents v5 | open | **open, unchanged** | `scripts/wire_study/wire_study.py:2-12` says v5 with 20 B / 28 B records; `FORMAT_VERSION = 8` at `binary_serializer.hpp:171` |
| L6 | `downsample_voxel_size` default described two ways | open | **open, unchanged** | `:787` 0.5; member `:3354` 0.0; `dscovox_multi_robot.launch.py:144` passes 0.1 and its comment (`:84-88`) names no default |
| L7 | `sdf_trunc` when the TSDF is off | resolved | resolved as written | memory `tsdf-off-does-not-zero-trunc` |
| S1 | `SCovoxNode` god object, the untested 30 % | PARTIAL | **partial → tier D** | no test target compiles `scovox_node.cpp` or `dscovox_node.cpp` (`CMakeLists.txt:53`, `:60` are their only users); D1–D5 are the remaining decomposition |
| S2 | `publishPointCloud` hardcodes two slots | FIXED | fixed | `static_assert` pair at `:2807-2808`; listed under "Not a smell" |
| S3 | envelope version an unnamed literal | FIXED | fixed | sender `:2503` and receiver `dscovox_node.cpp:314` read `BinarySerializer::ENVELOPE_VERSION` (`binary_serializer.hpp:179`) |
| S4 | codec header says revision 6 | FIXED | fixed | `binary_serializer.hpp:64-73` names no number |
| S5 | `0xFFFF` written 76 times unnamed | FIXED | fixed in `scovox`; replay residue → **F2** | `kEmptySlot` at `voxel.hpp:55`; `replay_scenenn.cpp` still spells three |
| S6 | `publishBinaryMap` repeats two idioms | FIXED | fixed | `node_utils.hpp:105` `emitSnapshotOrTouched`, `:144` `gateAndRefresh`; `test_publish_gate.cpp` 15 cases |
| S7 | `map_mtx_` a comment-enforced contract | FIXED | fixed | `map_lock.hpp`; `test_map_lock.cpp` 5 cases; the witness is listed under "Not a smell" |
| S8 | one build switch without a default | FIXED | fixed | `e0_counters.hpp:45-47` |
| S9 | `version.hpp` included by nothing | FIXED | fixed | `version.cpp` in `scovox_core/CMakeLists.txt:50` and the replay's `CMakeLists.txt:73`; the node logs `buildSwitches()` at `:277` |
| S10 | comments naming removed types | PARTIAL, watch | **unchanged, watch** | `SemDirMap` 8, `SemDirVoxel` 14, `Bresenham` 0: the review's counts |
| S11 | three warnings, two weak tests | FIXED | fixed | `split_memory_demo.cpp:17-18`; `test_beta_update.cpp:108-109`, `:116`, `:198-199` |
| errata | four commits do not compile | recorded, not fixed; decision pending | **still pending (G13)** | `f202860`, `eb4f9b3`, `8de5be1`, `bd45049` precede `f766258` in `git log`; `new_experiments` is 39 commits ahead of `origin`; `backup/review-series-pre-split` exists |
| round 3 | carve weight asymmetric; `s_total` counts looks | documented, nothing changed | documented; no action | both bound what G6 may read; neither is a fix |
| bench | three instrumentation defects | fixed | fixed as written | E-W20 removed the `absorbed` counter |
| E-W21/22 | carve-off and the deposit inventory | experiments | experiments; the one code candidate they name is **G14** | memory `sparse-branch-counters-cost` |

### Open

- [ ] **G1 — One source of parameter defaults (M1).** Nothing has landed:
  there is no `promotedParams()`, the four structs still carry their own
  numbers (`map_interface.hpp:28`, `sem_split_map.hpp:149`,
  `scovox_map_split.hpp:40`, `tsdf_map.hpp:48`), the replay has a fifth set
  in `Args` (`replay_scenenn.cpp:42`), and the node's `dp()` calls are a
  sixth (`:362-369`: `leaf_bits` 3, `dir_leaf_bits` 2, `w_occ` 2.0). *Decide*
  which struct is canonical, give it the promoted values, and make the others
  inherit from one factory so a divergence is a one-place diff. Do A3 (delete
  the inert `scovox::Params` fields) first; B1 (the `dp` lambda) is where the
  node side of this lands. Proof: the startup log (`:298`, `:342` print the
  effective values) is unchanged, and a wire-frame md5 from one bag is
  unchanged with every yaml as it is.

- [ ] **G2 — Declare the six library knobs as ROS parameters (M2).**
  `hit_flat_share`, `inc_mode`, `inc_thresh`, `class_evidence_saturation`,
  `ray_spread` (`SemSplitMap::Params`) and `far_voxel_fast_paths`
  (`ScovoxMapSplit::Params`) still have no `dp()`; the node prints two of
  them without declaring them (`:298`, `:342`), and
  `scovox_best_method.yaml:120` already records that
  `class_evidence_saturation: 0` needs the parameter to exist. Launch half:
  none of `scenenn_eval`, `scenenet_eval`, `semantickitti_eval` takes
  `semantic_band_length`; `w_occ` defaults 6.0 in two of them and `kappa0`
  2.0 is set per file. Declare the six with the struct defaults so nothing
  changes at default; take the eval launch defaults from one dict. Pairs with
  G5. Proof as G1.

- [ ] **G3 — dscovox: build the fused Dir grid at `dir_leaf_bits`; decide
  the first-pose cache (M5).** `dscovox_node.cpp:415-443` construct both
  fused grids from `P.leaf_bits`, while the sender's Dir grid uses
  `dir_leaf_bits` 2 (`:368`, `:97`). Memory shape only, but it is the shape
  the memory claims are made on. The cached `T_map_source` (`:98-102`) is
  still taken from the first update and never refreshed; the file banner
  (`:26-34`) calls that a correctness bug the moment c-slam is on. The
  fold-order half is done (`consensus_merge.hpp:141-150`). *Decide:* refresh
  the cache from each message's `map_from_source` (the review's fix), or keep
  the banner as the contract while c-slam stays off. Proof: fused output
  cloud md5 on one multi-robot bag with static TFs, identical before/after.

- [ ] **G4 — Reword the two block-run comments; the M6 hazard is
  withdrawn.** The review read `kBlockVoxels = 512` as a hard-coded
  `leaf_bits = 3` that a differently launched sender would break. Re-read
  2026-09-06: the encoder derives the block key from the coordinate itself
  (`binary_serializer.hpp:567-569`: locals by `& 7`, block by exact division)
  and the decoder reverses it (`:697-700`: `bx*8 + lx`); `leaf_bits` enters
  neither side, so a sender at leaf bits 1, 2 or 4 encodes and decodes the
  same coordinates and only the run density changes. No header field, no
  `FORMAT_VERSION` bump and no refusal is needed. What is wrong is the prose:
  `:20` and `:530` call the tile "their 8×8×8 Bonxai leaf block
  (`leaf_bits=3`)". Say "8×8×8 coordinate tile, independent of `leaf_bits`".
  Diff read only.

- [ ] **G5 — Separate the wire quantisation scale from the evidence cap;
  expose the class cap (M7).** Unchanged: `dp("evidence_saturation", 1000)`
  at `:404` feeds both the opt-in Beta cap and the wire `quant_step`, and
  `class_evidence_saturation` defaults −1 and follows it
  (`sem_split_map.hpp:239`, `sem_split_map.cpp:1130-1131`). No
  `wire_evidence_scale` exists. The three RGB-D yamls document the coupling;
  that is the "partly resolved". The structural fix stands: a
  `wire_evidence_scale` parameter for the codec, and `class_evidence_saturation`
  declared (G2). Proof: with both new parameters at their defaults the wire
  frames from one bag are byte-identical; with `evidence_saturation: 0` and a
  non-zero wire scale the payload stays u8 (the review's motivating case).

- [ ] **G6 — Publish the posterior marginal as `semantic_confidence`
  (M11).** `argmaxClassConfidence` still returns
  `(best_cnt + 1) / (sum_cnt + n_active + effectiveResidual(v) + 1)`
  (`node_utils.hpp:326-329`), the Laplace-plus-Hutter basis the calibration
  work rejected; the settled basis is `reparam()`'s marginal `alpha_best / A`.
  Consumers: `scovox_node.cpp:2925` (via the `SemBetaVoxel` projection at
  `:2921`) and `dscovox_node.cpp:706`. The review's own caveat is why this is
  a decision and not a tier-A item: nothing scored reads the field, so mIoU
  cannot move, but the wire value and the `vis_gate` labelling threshold do.
  Two mechanisms in the review's round-3 addendum bound what any replacement
  may read: `s_total` counts looks including null-class hits, and the band is
  un-batched (F6), so a raw count overstates evidence; the marginal is the
  reading that survives both. Proof: before/after histogram of the field and
  of the gate's pass rate on one bag; `.slots` md5 unchanged.

- [ ] **G7 — Fix the nine stale byte-size and type-name comments (L1).**
  All still stale on re-read; one moved. `sem_split_map.hpp:5` ("alternative
  to `SemDirMap`"), `:8` (`BetaVoxel` 8 B; shipped 4 B under
  `SCOVOX_BETA_U16=1`), `:10` (`DirVoxel` 16 B; shipped 20 B under
  `SCOVOX_TRACK_QMAX=1`); `sem_split_map.cpp:705` ("16 B DirVoxel", was
  `:690`); `dir_voxel.hpp:4` ("16-byte") and `:26` ("total: 16 B"), while
  `:99-100` and `:232-233` already give the correct account;
  `beta_voxel.hpp:17-20` ("free-space leaf-blocks at 8 B", "the 16 B
  `DirVoxel`"); `tsdf_voxel.hpp:11` (a `SemBetaVoxel` grid; the live pair is
  `BetaVoxel` ∥ `DirVoxel`); `tsdf_map.hpp:15-17`, `:49` (`SemBetaMap`, a type
  declared nowhere); `binary_serializer.hpp:40`, `:194` ("/ 65535" u16 step;
  the codec is u8 sqrt-companded). Under the standing rules the fix is to say
  which build the number describes, or to drop the number and point at the
  `static_assert` that pins it. Diff read only; goes with E1–E3.

- [ ] **G8 — Sweep the `.md` references in `scovox_slot_rules/scripts` (L2,
  scripts half).** 28 references in 17 files: `DESIGN.md` ×12 (no such file
  anywhere), `FINDINGS.md` ×9 and `PLAN.md` ×7 (both now under
  `scovox_slot_rules/archive/`). `admit_scenes.py` (5), `coverage_map.py` (3)
  and `scene_table.py` (3) carry the most; the other fourteen have one or
  two. `RESULTS.md`, `FEATURES.md`, `NEW_EXPERIMENTS_PLAN.md` and
  `results/aggregate_scenes.md` resolve and stay; `a.md`
  (`promoted_perscene.py:129`) is an attribute, not a file. Decide with E2
  whether the no-doc-pointer rule reaches scripts; if it does, delete the
  references rather than re-point them. Lives in `scovox_slot_rules`.

- [ ] **G9 — Decide the legacy substrate: ablation baseline or delete
  (L4).** Unchanged: `voxel.hpp`, `sembeta_voxel.hpp`, `scovoxmap.hpp/.cpp`
  and the NAIVE / MAJORITY_VOTE updaters are built, the node still routes
  `semantic_mode` to them (`:419-420`), and `test_beta_update` (32 cases) and
  `test_consensus` (25) exercise `scovox::Map`, which neither node maps with.
  Two of these are live and stay regardless: `Voxel` is the RPC projection
  type, and `SemBetaVoxel` is what the node projects into for the viz cloud
  (`:2921`) and what G6 reads. The candidates are `scovox::Map` and the two
  modes, which are also cleanup item 4 in the F table. If kept, one sentence
  in `semantics.hpp` saying they are the ablation baseline is the record; if
  deleted, the two test binaries go with them. Proof: build + `ros-test`
  green at the new case count.

- [ ] **G10 — Mark `wire_study.py` historical or port it to rev 8 (L5).**
  `scripts/wire_study/wire_study.py:2-12` still documents the v5 frame (20 B
  Beta, 28 B Dir records); the serializer is `FORMAT_VERSION = 8`
  (`binary_serializer.hpp:171`) with block runs and u8 companding, so the
  script's re-encodings do not start from the shipped bytes. A three-line
  banner is enough unless the study is to be re-run. Doc-only.

- [ ] **G11 — One description of the `downsample_voxel_size` default
  (L6).** `:787` says 0.5, the member initialiser `:3354` says 0.0
  (overwritten before use), `dscovox_multi_robot.launch.py:144` passes 0.1
  and its comment (`:84-88`) does not say what it overrides;
  `scovox_bin_min.yaml:72` also sets 0.1. Initialise the member to the same
  0.5 as `dp()` and have the launch comment state the code default. Goes in
  the A1–A4 commit.

- [ ] **G12 — Decide `FarCarveBitIdenticalToFullWalk` (M4).** Still the one
  red in the suite (183/184), band 0.30 arm only, pre-existing; not re-run
  this session. Not fixable from this list; it is a core-path decision: make
  the far-carve arm byte-identical at that band, or drop that band from the
  test with the reason in `REVIEW_LOG.md`. Until then Verification item 1
  has to keep carving it out, which weakens "suite green" as a gate for
  tiers A and B.

- [ ] **G13 — Decide the history rewrite before the next push (errata).**
  `f202860`, `eb4f9b3`, `8de5be1` and `bd45049` each call `gateAndRefresh` /
  `emitSnapshotOrTouched` before `f766258` defines them, so `git bisect`
  cannot build across four commits. The branch is still 39 commits ahead of
  `origin/new_experiments`, so the rewrite (split `f766258`, rebase) is still
  cheap, and `backup/review-series-pre-split` at `33724d4` is in place. Every
  commit this list adds on top makes the rebase larger, which is why it is
  step 0 in the commit sequence. Either outcome is fine; "accept and note
  it" is a one-line `REVIEW_LOG.md` entry. Proof if rewritten:
  `git bisect run` builds at every commit in the range.

- [ ] **G14 — Compile out the sparse-branch counters: landed as `a062908`
  (2026-09-06 14:59); three of its five proof items remain.** The review's
  closing paragraph names the locked atomic branch counters inside
  `sparse_add_class` (four `std::atomic<uint64_t>` globals, about 14
  `lock`-prefixed RMWs per ray, read only by diagnostics) as the top
  byte-identical candidate on the deposit path. `a062908` does it:
  `voxel.hpp` gains `SCOVOX_SPARSE_BRANCH_COUNTERS` (default 0, forced 1
  under `SCOVOX_E0_COUNTERS`, `:92-97`) and a `SCOVOX_SPARSE_BUMP` macro
  (`:100-104`); the nine bump sites (`voxel.hpp:172,179,221,224`,
  `dir_voxel.hpp:356,366,378,430,435`) go behind it; the node warns at
  startup when `eviction_stats_csv` would be all zeros (`:840-846`) and says
  "not counted in this build" at shutdown instead of printing zeros
  (`:3426-3433`). Not started by this list and not reviewed here beyond a
  diff read. Against the five things "done" needed: (1) `buildSwitches()`
  reporting the switch — **done 2026-09-07** with G15; a default build
  prints `SPARSE_BRANCH_COUNTERS=0`. (2) An E0 build showing the
  `dumpEvictStats()` cross-check still counts — `REVIEW_LOG.md` E-W24 asserts
  it (`:3657`) without citing a build. (3) Byte-identity — done: E-W24 has
  the counters-on build md5-identical to the E-W22 control binary, and scene
  016 at full length dumping byte-identically on and off in both the shipped
  and the carve-off configs. (4) The timing number — measured on scene 016
  only: 18 paired reps, +2.31 ms/frame (95 % CI +0.62 to +4.00, p 0.0208,
  faster in 13/18), 1.45 % of the shipped frame; the 8-scene run with an
  inert control was dropped on instruction (E-W24, `:3665`), so it stays a
  one-scene number. (5) The E8.7 harnesses built with the flag set for the
  whole build — not evidenced. Open for (1), (2) and (5). The numbers live in
  `REVIEW_LOG.md` E-W24 and `scovox_code_structure.md` §4.6, not in a
  comment.

- [x] **G15 — `buildSwitches()` names seven of the nine live switches.**
  **Closed 2026-09-07.** All ten live switches are now on the line; see the
  addendum below for what was done and what proved it.
  S9 (`db659af`) made `version.cpp` the binary's self-report under the
  contract that every switch with an `#ifndef` default appears on one
  `NAME=value` line, because a mistyped `-D` preprocesses to nothing and an
  md5 cannot say which switch moved. Two switches were added after S9 and
  neither is on the line: `SCOVOX_WALK_MARGIN_VOX`
  (`scovox_map_split.hpp:349-352`, from `703eb7b`) and
  `SCOVOX_SPARSE_BRANCH_COUNTERS` (`voxel.hpp:92-97`, from `a062908`).
  `version.cpp:41-50` still lists seven plus `NDEBUG`, and the `version.hpp`
  header comment still says "the seven switches". The line is already
  logged where it should be — the node prints it once at startup
  (`scovox_node.cpp:277`), and `replay_scenenn.cpp:428` and
  `replay_kitti.cpp:260` print it to stderr — so today a `margin0` timing
  probe and a counters-on build both report the same line as the shipped
  default, which is the hazard the function exists to close.
  *Change:* append the two pairs in `version.cpp` (the margin macro is
  declared inside `scovox_map_split.hpp`; either include it there or move
  the `#ifndef` block to `voxel.hpp` next to `SCOVOX_K_TOP`), and correct
  the header comment's count. *Effect:* log line only; the function returns
  a string literal, so no map byte can move. The audit for future drift is
  `grep -rn '^#ifndef SCOVOX_' src/scovox_core/include` against the
  literal in `version.cpp`; the two should agree minus the include guards.

- [ ] **G16 — a `WALKER_TIMERS=0` build publishes `tsdf_ms=0.0
  sembeta_ms=0.0` and says nothing.** New 2026-09-07, created by the change
  that closed G15. `SCOVOX_WALKER_TIMERS` (`walker_timers.hpp:28`, default 1)
  compiles the walkers' ten `steady_clock::now()` reads away; at 0 both
  `tsdfTimeUs()` and `semdirTimeUs()` return 0 for the life of the process.
  The node reads them unconditionally at `:1331-1332` into `ScanTailStats`
  and prints them in both per-frame log lines (`:1409-1411`, `:1899-1902`).
  Those two format strings are a **frozen parser contract** (`:1283-1290`),
  so the fields cannot simply be dropped — and `0.0` there is
  indistinguishable from "the walk was instant" to any parser reading them.
  *Change:* the precedent already exists one tier up — G14's landed commit
  warns once at startup when `eviction_stats_csv` would be all zeros
  (`:840-846`). Do the same: `#if !SCOVOX_WALKER_TIMERS`, one
  `RCLCPP_WARN` at construction naming the two fields and the switch.
  *Effect:* one startup log line on a non-default build; no field, no
  format string and no map byte moves. *Why it is not urgent:* the default
  is 1 and the build banner already prints `WALKER_TIMERS=` on the same
  startup log (`:277`), so the information is present — it is one line
  away from the number it explains rather than beside it. Measured worth
  of the 0 build on the carve-off replay: 1.102x [1.096, 1.109] (E-W30).

**Watch, no action.** S10 (removed-type names in comments; counts unchanged
at 8 / 14 / 0) stays a watch item. H4 is an experiment-ledger item and is
tracked in memory, not here.

### Closed on evidence

- **S2, S3, S4, S6, S7, S8, S9, S11 — fixed and still in the tree.** Each
  was re-read at HEAD (table above). Two are now load-bearing for this list:
  S7's lock witness is what C1 asks for more of, and S6's helpers are where
  D4 adds. None is re-opened.
- **M6's stated impact is withdrawn; its comment half is G4.** Both codec
  sides derive the tile from the coordinate, so no launch can produce the
  mismatch the review described.
- **M5's fold-order half is done.** `consensus_merge.hpp:141-150` states the
  dependence and the fixed source order.
- **Cross-references.** M3 → F1 (one decision, one commit). M9 → F6
  (deferred experiment; run on request). S5 → F2 (replay residue). S1 →
  D1–D5 (the remaining decomposition). L4 ↔ cleanup item 4 → G9. The
  round-3 mechanisms (carve-weight asymmetry between `integrateRay`'s two
  branches; `s_total` counts looks) are documented facts with no action; G6
  cites them.

---

## Not a smell — leave alone

These are correct and a later cleanup should not "simplify" them:

- The lock-witness pattern (`MapReadLock`/`MapWriteLock`, helpers taking
  `const MapLockHeld&` / `MapWriteHeld&`). C1 asks for *more* of it.
- `scheduleMemUsage` (`:1902`): owned `std::thread`, CAS single-in-flight
  guard, joined in the destructor. The 97 lines are the cost of doing this
  right.
- The PointCloud2 buffer validation before any `reinterpret_cast`, the
  `K_TOP` `static_assert`s in `publishPointCloud`, and the `memcpy` bit-cast
  for the packed `rgb` field.
- `finishScanTail` / `scanLogEpilogue`: the shared tail that already removed
  the RGB-D/LiDAR duplication once.
- The zero-timeout TF lookup in `publishBinaryMap` (`:2273-2275`) and its
  defer-and-retry; C1 copies it.
- The helpers already in `node_utils.hpp` (`forEachCellInBox`,
  `emitSnapshotOrTouched`, `gateAndRefresh`, `heartbeatReemit`,
  `generateSemanticColors`). D4 adds to that file rather than starting
  another.

---

## Verification

No test instantiates `SCovoxNode`, and the mapper replay in
`scovox_slot_rules` links `scovox_core` only, so neither `./dev.sh test` nor
`pristine_head_replay.sh` can see any of this file. The proof for each tier:

1. **A, B, E:** `./dev.sh ros-test` green (143 `scovox_mapping` cases at
   HEAD; the one known core failure `FarCarveBitIdenticalToFullWalk` is
   pre-existing and unrelated).
2. **C:** the item's stated effect is the only observable change. For C1,
   a bag run with `mode: rolling` must show no `skipping rolling crop`
   warnings after the first second. For C2/C3, attach a subscriber late and
   confirm one frame arrives.
3. **D:** record the `/scovox/bin` wire frames (and the `[tsdf_dump]` file
   if `tsdf_dump` is on) from one bag on `703eb7b` and on the branch, same
   yaml, same bag, same `--clock` rate; md5 per frame must match. Any
   mismatch fails the item regardless of how small.
4. **Order:** land D8 (format-only) either first or last, never mixed with
   another item, so every other diff stays readable.
5. **F:** F2 is inside the replay's reach: `pristine_head_replay.sh` on one
   scene, `.slots` md5 equal. F1 under option (a) is a parameter rename, so
   the proof is the launch files and yamls in the same commit plus a node
   startup log showing the cull still engages; under option (b) it is a new
   deposit weight and needs the full 8-scene A/B before it ships. F3 is a
   doc edit with no proof beyond a read.
6. **G:** G4, G7, G10 are comment edits (diff read); G11 goes with tier
   A. G1, G2, G5 change the parameter surface: the startup log (`:298`,
   `:342`) shows the same effective values and a wire-frame md5 from one
   bag is unchanged with every new parameter at its default. G3: fused
   output md5 on one multi-robot bag with static TFs. G6 changes a wire
   value by design: before/after on the `vis_gate` pass rate, `.slots` md5
   unchanged. G9, if deleting: build + `ros-test` green at the new case
   count. G12 moves the suite to 184/184, or to 183/183 with the reason
   logged. G13, if rewritten: `git bisect run` builds across the range.
   G14 landed; what remains of its proof is listed in the item. **G15 is
   done (2026-09-07):** the replay's `build:` stderr line now carries all
   ten pairs including `SPARSE_BRANCH_COUNTERS=0` and
   `WALK_MARGIN_VOX=2.7320508f`; the four 400-frame dump gates were re-run
   after it and are unchanged, as a `.rodata` literal cannot move a map
   byte. The standing audit is `grep -rn '^#ifndef SCOVOX_'
   src/scovox_core/include` against that line, guards excluded. G16: an
   `--ros-args` run of a `WALKER_TIMERS=0` build showing whatever the
   decision is — either the two fields absent, or a startup warning that
   they will read `0.00`.

## Suggested commit sequence

0. G13 first, while the rebase is still cheap; nothing below should land
   on top of the four non-compiling commits before that decision.
1. A1–A4 + E4 + G11 (one commit: "remove dead parameters and vestigial
   returns").
2. B1–B3, B8 (one commit: "hoist dp, frameId, alias").
3. B4 (own commit: the silent catch is the one behaviour change in tier B).
4. B6, B7 + C3 (one commit: cloud schema and zero-crossing publisher).
5. C2 + the `PublishGate` type, then C4, then C5, then C1 (each its own
   commit; C1 last because it is the one with a startup-visible effect).
   F1 goes here too once M3's option is chosen. Then G1, G2, G5 as one
   parameter-surface commit; G3 and G6 each on their own.
6. B5, then D2–D5 one at a time, each with its wire-frame md5 check.
7. D6, D7, E1–E3, G4, G7.
8. D8, formatting only.
9. In `scovox_slot_rules`, independently: F2 (one commit), F3 (banner),
   G8, G10.
10. Own commits, when their decision is made: G9 (with its tests), G12
    (core path). **G15 is done** — it landed with the E-W30 walker work,
    uncommitted at the time of writing; the audit in §Verification is what
    keeps it from drifting when the next build-switch is added. G14 landed
    as `a062908`; its open proof items need no further commit. G16 belongs
    with C5 in step 5, since both are about a field the node publishes
    without the state to back it.
