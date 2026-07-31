# Literature review — significance-gated Bayesian map sharing

**Date:** 2026-07-31
**Status:** Synthesis. Merges `review-01.md` and `review-02.md` (two independent
responses to [lit_review_prompt_2026_07_31.md](lit_review_prompt_2026_07_31.md))
against [comms_design_2026_07_30.md](../design/comms_design_2026_07_30.md), with
an independent verification pass over the load-bearing citations.
**Provenance rule used here:** a claim appears below only if it survived
re-checking. Where the two source reviews disagree, the disagreement is resolved
and the resolution is recorded in Appendix A. Where neither could be confirmed,
the entry is marked and kept out of the argument.

---

## 0. Headline findings

Five things changed — four from the verification pass, the fifth from reading
the merge code after it. Read these before the sections that follow; three of
them alter what the paper should say.

**0.1 — The primary contribution survives contact with the literature.** No
system was found that reduces volumetric map-sharing bandwidth while leaving the
receiver a fusable posterior. Every candidate checked degrades what the peer
holds: to a binary classification, to a coarser resolution, or to learned
features. The claim as stated in the design doc is defensible.

**0.2 — The #1 open threat is resolved, and it resolves *in the paper's
favour*.** Łuczyński et al. (OCEANS 2017) was flagged in all three source
documents as the likeliest place an aggressive per-cell gate already exists. Full
text obtained and read. It is a **state-flip gate**, not a continuous one — §VI.A
states the rule directly:

> "When integrating the point cloud into the map, it is checked if any voxel has
> changed its state, i.e., if a previously unknown region is now identified as
> free/occupied, a voxel that was occupied is now free or the free voxel is now
> occupied."

and, decisively for our argument, immediately after:

> "At the same time the log odds in the OctoMap are constantly updated on board
> of the vehicle."

That is the exact pathology this paper is about — continuous belief evolution
maintained locally and never transmitted — stated by the prior art itself.
Łuczyński is now a *supporting* citation, not a threat. Design doc open
question 5 can be closed.

**0.3 — A factual error is carried by all three source documents and must not
reach a submission.** Łuczyński et al. is **not** an acoustic-link paper and
does not involve ~100 kbps acoustic links. It is a Ku-band **satellite** link
for ROV teleoperation (DexROV): 768 kb/s uplink, 256 kb/s downlink, 620 ms
nominal RTT (§II). It is also **single-vehicle telemetry** (ROV → vessel →
onshore mission control), not multi-robot map sharing, and its evaluation is
**simulation only** (§VII, Docker + `netem`). Review-01 additionally asserts it
"gates on bounding boxes of changed discrete states" — the bounding box is a
*resolution and colour* selector for an operator-specified region of interest,
not the transmission gate. Every one of these characterisations needs fixing
where it appears.

Consequence: **underwater acoustic map transmission remains genuinely unswept.**
Resolving Łuczyński did not close that subfield, because Łuczyński was never in
it. This is now a smaller but still-open gap.

**0.4 — The source reviews' citation sets are not safe to use as-is.** Both
contain misattributions; review-01 contains at least one entry that does not
appear to exist as described. Details in Appendix A. Nothing below inherits a
citation that failed re-checking.

**0.5 — (post-review code read) The data-incest threat is closed by the
architecture, and the scope argument upgrades from assertion to proof.** The
merger keeps a replica grid pair per source robot and rebuilds every touched
fused cell from scratch — reset to prior, fold all sources' current values
(`refoldBeta`,
[dscovox_consensus.hpp:163](../../src/scovox_mapping/include/scovox/dscovox_consensus.hpp#L163)).
The fused cell is a pure function of the current source replicas, no carve
posterior ever ingests fused output, and therefore the gate provably cannot
change what fusion produces — only *when* a replica attains a value. Details and
the two stated preconditions (acyclic topology; the K_TOP semantic
re-truncation, which is the full paper's hypothesis under test, not a defect) in
T2, which is now marked closed.

---

## 1. Verified citation set

Grouped by the related-work paragraph each entry serves. Verification tags:

- `[FULL-TEXT]` — full text obtained and read this pass; load-bearing claims quoted
- `[VERIFIED]` — bibliographic record retrieved and confirmed this pass
- `[DESIGN-DOC]` — carried from the design doc's reference list, which was
  independently accurate everywhere it was spot-checked; bibliographic data not
  re-retrieved this pass
- `[UNVERIFIED]` — second-hand only; source of the claim named

DOIs appear only where confirmed. A blank DOI is a lookup task, never a guess.

### 1.1 The conjugate volumetric substrate (¶1)

- **Doherty, K., Wang, J., Englot, B.** "Bayesian Generalized Kernel Inference for
  Occupancy Map Prediction." *ICRA*, 2017, pp. 3118–3124. `[VERIFIED]`
  Role: the Beta occupancy voxel. The substrate this paper transmits.
  DOI unconfirmed — review-01 gives `10.1109/ICRA.2017.7989356`; plausible, check.
- **Gan, L., Zhang, R., Grizzle, J., Eustice, R., Ghaffari, M.** "Bayesian Spatial
  Kernel Smoothing for Scalable Dense Semantic Mapping." *IEEE RA-L* 5(2), 2020.
  `[DESIGN-DOC]` Role: the Dirichlet semantic counting model.
  DOI unconfirmed — review-01 gives `10.1109/LRA.2020.2965390`; plausible, check.
- **Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C., Burgard, W.**
  "OctoMap: An efficient probabilistic 3D mapping framework based on octrees."
  *Autonomous Robots* 34(3):189–206, 2013. DOI `10.1007/s10514-012-9321-0`.
  `[DESIGN-DOC]` Role: the state-flip baseline (`resetChangeDetection`), and the
  clamping policy that bounds the evidence arm's useful range.
- **Chen, G., Dong, W., Peng, P., Alonso-Mora, J., Zhu, X.** "Continuous Occupancy
  Mapping in Dynamic Environments Using Particles." *IEEE T-RO* 40:64–84, 2024.
  arXiv:2202.06273. `[VERIFIED]` Role: optional — argues continuous occupancy
  representations against grid discretisation. **Note:** "DSP-Map" is the system
  name, not the title; review-01's title, author and year are all wrong (see A.2).

### 1.2 Event-triggered estimation — the mechanism (¶5)

This cluster is where the paper concedes priority. It should be cited generously.

- **Miśkowicz, M.** "Send-On-Delta Concept: An Event-Based Data Reporting
  Strategy." *Sensors* 6(1):49–63, 2006. DOI `10.3390/s6010049`. `[DESIGN-DOC]`
  Role: the canonical citation for the rule. Cite first, concede immediately.
- **Åström, K. J., Bernhardsson, B.** "Comparison of Riemann and Lebesgue sampling
  for first order stochastic systems." *CDC*, 2002. `[DESIGN-DOC]`
  Role: why gate at all.
- **Årzén, K.-E.** "A simple event-based PID controller." *IFAC World Congress*,
  1999. `[DESIGN-DOC]` Role: deadband + max-period safety timer — the heartbeat
  precedent. Cite for the liveness arm, which is otherwise indefensible as novel.
- **Trimpe, S., Campi, M.** "On the Choice of the Event Trigger in Event-based
  Estimation." *EBCCSP*, 2015. `[DESIGN-DOC]`
  Role: **structurally load-bearing.** Eq. 5's two-term KL expansion (mean-shift +
  log-precision-ratio) is what makes the two-arm rule a principled decomposition
  rather than two heuristics. Verify the equation number against the PDF before
  submission — the whole "not stapled together" argument rests on it.
- **Trimpe, S., D'Andrea, R.** "Event-Based State Estimation With Variance-Based
  Triggering." *IEEE TAC* 59(12):3266–3281, 2014. `[DESIGN-DOC]`
  Role: the dual of the evidence arm — fires when variance *grows* under process
  noise; a static map has none, so we fire when it *shrinks*. State the asymmetry.
- **Han, D., Mo, Y., Wu, J., Weerakkody, S., Sinopoli, B., Shi, L.** "Stochastic
  Event-Triggered Sensor Schedule for Remote State Estimation." *IEEE TAC*
  60(10):2661–2675, 2015. DOI `10.1109/TAC.2015.2406975`. arXiv:1402.0599.
  `[VERIFIED]` Role: **deterministic deadbands destroy the closed form; stochastic
  triggers were invented to avoid it.** This is the censored-posterior citation.
  Do not confuse with the distinct "Multi-Sensor Scheduling for State Estimation
  with Event-Based, Stochastic Triggers" (arXiv:1502.03068) — review-01 merged the
  two (see A.3).
- **Battistelli, G., Chisci, L., Gao, L., Selvi, D.** "Event-triggered distributed
  Bayes filter." arXiv:1902.09825, 2019. `[DESIGN-DOC]`
  Role: closest architectural match — KL against a reference density that *is* the
  last transmission. The single most dangerous citation for the novelty claim;
  cite it before a reviewer does.
- **Mitra, A., Richards, J., Bagchi, S., Sundaram, S.** "Event-Triggered Distributed
  Inference." *CDC*, 2020. `[DESIGN-DOC]`
  Role: relative/multiplicative belief trigger with a.s. consistency and an
  exponential rate — the proof template for "gating does not break convergence."
- **Marck, J. W., Sijs, J.** "Relevant sampling applied to event-based
  state-estimation." *SENSORCOMM*, 2010. `[UNVERIFIED]` — cited by the design doc
  as "Marck & Sijs 2010", by review-02 as an *Automatica* 2016 paper with Sijs
  first. Two different works may be conflated; resolve before citing.
- **Sijs, J., Noack, B., Hanebeck, U.** "Event-based state estimation with negative
  information." *FUSION*, 2013. `[DESIGN-DOC]`
  Role: silence carries information. Cite in the negative-information ablation.
- **Rago, R., Willett, P., Bar-Shalom, Y.** "Censoring sensors: a
  low-communication-rate scheme for distributed detection." *IEEE TAES*
  32(2):554–568, 1996. `[DESIGN-DOC]`
  Role: the optimality argument for threshold-shaped gates, and the name of the
  pattern Rocha belongs to.

### 1.3 Timeliness metrics (¶5, and the evaluation section)

- **Maatouk, A., Kriouile, S., Assaad, M., Ephremides, A.** "The Age of Incorrect
  Information: A New Performance Metric for Status Updates." *IEEE/ACM Trans.
  Networking* 28(5):2215–2228, 2020. DOI `10.1109/TNET.2020.3005549`.
  arXiv:1907.06604. `[VERIFIED]`
  Role: **"peer lag" is this metric.** Adopt the name (§4, T3).
  Note: review-01's DOI `10.1109/TNET.2020.2990202` is wrong (see A.1).
- **Yates, R. D., Sun, Y., Brown, D. R., Kaul, S. K., Modiano, E., Ulukus, S.**
  "Age of Information: An Introduction and Survey." *IEEE JSAC* 39(5), 2021.
  arXiv:2007.08564. `[VERIFIED — arXiv record]` Role: the AoI framing and
  vocabulary; cite for context, AoII for the actual metric.

### 1.4 Deployed volumetric map sharing — the differentiator cluster (¶3)

This is the most important cluster in the paper. Every entry is evidence for
"they degrade the posterior."

- **Łuczyński, T., Fromm, T., Govindaraj, S., Mueller, C. A., Birk, A.** "3D Grid
  Map Transmission for Underwater Mapping and Visualization under Bandwidth
  Constraints." *OCEANS 2017 – Anchorage*, 2017. IEEE Xplore doc 8232281.
  `[FULL-TEXT]`
  Role: **state-flip gate under a hard bandwidth budget, with the posterior
  explicitly retained on-vehicle and not transmitted.** Quote §VI.A (both lines in
  §0.2). Correct framing: satellite teleoperation, single vehicle, simulation-only
  evaluation. Bandwidth: full-map retransmission reaches ~700 KB/s at 10 cm and
  ~330 KB/s at 15 cm (Figs. 5, 7); their incremental scheme drops to single-digit
  KB/s (Fig. 6).
- **Ohradzansky, J., et al.** "Multi-Agent Autonomy: Advancements and Challenges in
  Subterranean Exploration." *Field Robotics*, 2022. arXiv:2110.04390. Code:
  `github.com/dan-riley/marble_mapping`. `[DESIGN-DOC]`
  Role: MARBLE — diff tree carrying full `getLogOdds()`, then serialised via
  `binaryMapToMsg`, 1 bit per leaf. ~100× reduction, binary receiver state.
  **Read the code, not the paper**, for the wire format claim.
- **FZI `vdb_mapping` / `vdb_mapping_ros2`.** `[DESIGN-DOC]`
  Role: the direct architectural analogue, and the strongest single sentence in
  the entire related-work section — its own README concedes the low-bandwidth mode
  *"has the downside, that the probabilistic framework is lost on the remote
  side."* Verify the quote against the current README and cite with a commit hash.
- **Reijgwart, V., Cadena, C., Siegwart, R., Ott, L.** "Efficient volumetric mapping
  of multi-scale environments using wavelet-based compression." *RSS*, 2023.
  `[DESIGN-DOC]` Role: saturated-region skipping and coefficient thresholding —
  compression-side prior art, and the nearest thing to "skip what is settled."
- **US 11,312,379** (octree map synchronisation). `[UNVERIFIED]`
  Role: cited by the design doc as quantising child nodes to a single bit.
  Re-verify against primary claim text before relying on it.

### 1.5 Collaborative SLAM communication budgets (¶2)

What a robotics reviewer will benchmark against. Note that all of these gate at
*graph* granularity — none transmits a dense posterior at all, which is both the
paper's opening and its evaluation problem (§5).

- **Tian, Y., Chang, Y., Herrera Arias, F., Nieto-Granda, C., How, J. P., Carlone,
  L.** "Kimera-Multi: Robust, Distributed, Dense Metric-Semantic SLAM for
  Multi-Robot Systems." *IEEE T-RO*, 2022. arXiv:2106.14386. IEEE doc 9686955.
  `[VERIFIED]` **Venue is T-RO, not RA-L 2021; first author is Tian, not Rosinol**
  (Rosinol is single-robot Kimera). Review-02 has both wrong (see A.4).
- **Lajoie, P.-Y., Beltrame, G.** "Swarm-SLAM: Sparse Decentralized Collaborative
  Simultaneous Localization and Mapping Framework for Multi-Robot Systems."
  *IEEE RA-L* 9(1):475–482, 2024. DOI `10.1109/LRA.2023.3333742`. arXiv:2301.06230.
  `[VERIFIED]` Role: inter-robot loop-closure prioritisation to cut communication.
  **Not Cieslewski** (review-02, see A.4).
- **Lajoie, P.-Y., Ramtoula, B., Chang, Y., Carlone, L., Beltrame, G.** "DOOR-SLAM:
  Distributed, Online, and Outlier Resilient SLAM for Robotic Teams." *IEEE RA-L*
  5(2), 2020. DOI `10.1109/LRA.2020.2967681`. `[VERIFIED — record confirmed]`
  Role: the standard low-bandwidth distributed SLAM benchmark.
- **Huang, Y., Shan, T., Chen, F., Englot, B.** "DiSCo-SLAM: Distributed Scan
  Context-Enabled Multi-Robot LiDAR SLAM with Two-Stage Global-Local Graph
  Optimization." *IEEE RA-L* 7(2):1150–1157, 2022. DOI `10.1109/LRA.2021.3138156`.
  `[VERIFIED]` **Not Wang** (review-02, see A.4).
- **Hermes**, "Streamlining Data Transfer in Collaborative SLAM Through
  Bandwidth-Aware Map Distillation." *IEEE RA-L*, 2025. IEEE doc 10918818.
  `[UNVERIFIED — review-02, no author list]`
  Role: potentially the closest recent baseline (entropy-gain submap distillation).
  **Highest-value verification task in this document** after the patents: get
  authors, DOI, and the actual selection rule.
- **Alice-SLAM**, "Accurate and Lite-Communication Collaborative SLAM for
  Resource-Constrained Multi-Agent." *IEEE RA-L*, 2025. IEEE doc 11207676.
  `[UNVERIFIED — review-02, no author list]`
- **CCMD-SLAM**, "Communication-Efficient Centralized Multirobot Dense SLAM With
  Real-Time Point Cloud Maintenance." 2024. IEEE doc 10530544.
  `[UNVERIFIED — review-02; venue given as "IEEE T-IT / Sensors", which is
  incoherent and signals the record was not retrieved]`

### 1.6 Learned collaborative perception (¶4)

- **Hu, Y., Fang, S., Lei, Z., Zhong, Y., Chen, S.** "Where2comm:
  Communication-Efficient Collaborative Perception via Spatial Confidence Maps."
  *NeurIPS*, 2022. arXiv:2209.12836. `[VERIFIED]`
  Role: per-cell gating, but on **absolute current-frame confidence with no
  temporal reference**, shipping features that cannot be Bayes-fused. The cleanest
  contrast in the paper: same granularity, opposite reference state.
- **Xu, J., Zhang, Y., Cai, Z., Huang, D.** "CoSDH: Communication-Efficient
  Collaborative Perception via Supply-Demand Awareness and Intermediate-Late
  Hybridization." *CVPR*, 2025, pp. 6834–6843. arXiv:2503.03430. `[VERIFIED]`
  Role: recency anchor; documents that learned feature compression degrades at
  ultra-low bandwidth and needs hybrid fusion. **First author is Junhao Xu**
  — review-01's "Derrick Xu" is wrong.
- **InfoCom**, "Kilobyte-Scale Communication-Efficient Collaborative Perception with
  Information Bottleneck." *AAAI*, **2026**. `[UNVERIFIED — record exists, authors
  not confirmed]` **Review-01's entry for this is wrong in title, venue-year, URL
  and authors** (see A.5). If cited, cite the real one, and note it is contemporary
  with rather than prior to this submission.

### 1.7 Decentralized fusion and data incest (¶6)

- **Julier, S. J., Uhlmann, J. K.** "A non-divergent estimation algorithm in the
  presence of unknown correlations." *ACC*, 1997. DOI `10.1109/ACC.1997.609105`.
  `[VERIFIED — record confirmed]` Role: Covariance Intersection, the canonical
  conservative-fusion answer to unknown correlation.
- **Generalized Covariance Intersection / Chernoff fusion for exponential
  families.** `[GAP — no single citation confirmed]`
  The GCI literature is real and well-established for random-finite-set and
  Gaussian posteriors (GCI minimises the weighted sum of KL divergences from the
  local posteriors, which is exactly the double-counting-avoidance property
  needed). **A Beta/Dirichlet-specific GCI treatment was not confirmed to exist.**
  Review-01's citation for this — Gao et al., *IEEE SPL* 2022, given the title
  "Distributed GGIW-CPHD-based extended target tracking over a sensor network" —
  has a title about extended-target tracking that does not match the claimed role
  of "derives GCI for Beta-distributed detection probabilities." **Do not cite it
  for that claim.** The nearest genuine thread is the RFS work on CPHD filtering
  with unknown probability of detection, which does carry Beta-distributed
  detection parameters; that thread needs a real search before the paper leans on
  it. Until then, treat the counting-family case as *open*, which is the honest
  and more useful position anyway (§4, T2).

### 1.8 Adaptive precision and suppression semantics (¶6, future work)

- **Olston, C., Jiang, J., Widom, J.** "Adaptive Filters for Continuous Queries over
  Distributed Data Streams." *SIGMOD*, 2003; and **Olston, C., Loo, B. T., Widom,
  J.** "Adaptive Precision Setting for Cached Approximate Values." *SIGMOD*, 2001.
  `[DESIGN-DOC]` Role: **the fixed global τ is explicitly their naive baseline.**
  Their contribution is adaptive per-object precision allocation. Cite honestly:
  this is why adaptive τ is future work rather than a claimed contribution.
- **Silberstein, A., Puggioni, G., Gelfand, A., Munagala, K., Yang, J.**
  "Suppression and Failures in Sensor Networks: A Bayesian Approach." *VLDB*, 2007.
  `[DESIGN-DOC]` Role: suppression vs. packet loss ambiguity — the direct
  justification for the liveness arm on a lossy radio.

### 1.9 Industrial and standards prior art (¶5 concession)

Cite these as a group, in one sentence, to establish that the mechanism is
unclaimable. Do not spend a paragraph.

- **OPC UA Part 4 §7.22.2** `DataChangeFilter` (`AbsoluteDeadband`,
  `PercentDeadband`). `[DESIGN-DOC]` Note the "last cached value = last value
  pushed to the queue" semantics — the last-*transmitted* reference state, in a
  1990s industrial standard.
- **BACnet Change-of-Value** (ASHRAE 135). `[DESIGN-DOC]`
- **AVEVA/OSIsoft PI** exception and compression (`ExcMax`). `[DESIGN-DOC]`
- **IEEE 1278 (DIS) dead reckoning**, 1993. `[DESIGN-DOC]`
- **ETSI TR 103 562**, Collective Perception Service object-inclusion rules.
  `[DESIGN-DOC]` A deployed per-object send-on-delta gate against last-transmitted
  state, with a 1 s liveness rule.

### 1.10 The nearest conceptual ancestor (¶3/¶5)

- **Rocha, R., Dias, J., Carvalho, A.** "Cooperative multi-robot systems: A study of
  vision-based 3-D mapping using information theory." *Robotics and Autonomous
  Systems* 53(3–4):282–311, 2005. `[DESIGN-DOC — full text read, per design doc]`
  Role: the closest conceptual ancestor. The design doc's three-way separation
  (per-measurement not per-voxel, Eq. 64/68; peer-memoryless; ships measurements
  not map state) is well argued and is retained here unchanged. Both source
  reviews independently reached the same conclusion, which is mild corroboration.

### 1.11 Patents

State factually. No infringement or validity opinion, in the paper or anywhere.

- **US 12,236,779 B2**, "Collective perception service enhancements in intelligent
  transport systems." Intel Corp. Granted **2025-02-25**; status **active**;
  expires 2041-10-15. `[VERIFIED — Google Patents record]`
  Claim 1 selects at **layer** granularity for layered cost-map data, not per cell.
  Confirms the design doc's prior.
- **US 2023/0110467 A1**, Intel, differential cost-map transmission of cells changed
  beyond a threshold versus previously transmitted messages; believed abandoned.
  `[UNVERIFIED — not re-retrieved this pass]` **Still the highest-severity
  unverified item in the document** — it is the only cell-granularity, versus-last-
  transmitted disclosure known. Get the primary claim text and the file wrapper.
- **Non-English filings (JP/CN/KR/DE):** still not searched. Unchanged gap.

---

## 2. Closest-work table

Ordered by how likely a reviewer is to raise it. The two columns that carry the
contribution are **reference state** and **receiver holds**.

| Work | What is gated / compressed | Granularity | Reference state | Receiver holds | They do A, we do B |
|---|---|---|---|---|---|
| **OctoMap** change detection (Hornung 2013) | Voxels whose binary classification flipped | Voxel | Last transmitted (`resetChangeDetection`) | Binary occupancy, or log-odds at full cost | Gates on a p = 0.5 crossing; we gate on graded divergence, so the peer sees confidence, not just class. |
| **Łuczyński et al. 2017** `[full text]` | Voxels that changed state (unknown→free/occ, occ→free, free→occ) | Voxel, plus an ROI bbox selecting resolution and colour | Last transmitted map state | Coarse- or fine-resolution voxel lists; classification only — **log-odds stay on the vehicle by design** | Identical gate topology under a hard budget, but discrete; they say plainly the posterior is not transmitted. Our gate ships it. |
| **MARBLE** (Ohradzansky 2022) | Diff tree of changed leaves, then `binaryMapToMsg` | Voxel | Last transmitted diff | **1 bit per leaf** | ~100× reduction by destroying the posterior; we target comparable bandwidth while preserving it. |
| **FZI `vdb_mapping`** | Voxels in an "overwrites" grid | Voxel / grid patch | Last updated | `bool` grid — README concedes the probabilistic framework "is lost on the remote side" | The same architecture, with the loss stated as a known downside; we remove the downside. |
| **Where2comm** (Hu 2022) | Learned BEV feature patches | Spatial cell (dense) | **None** — absolute current-frame confidence, memoryless | Learned features; no explicit posterior | Same granularity, *no temporal reference*: they ask "is this important now?", we ask "has the peer drifted from me?" |
| **CoSDH** (Xu 2025) | Learned features under supply-demand awareness | Spatial region | Peer's requested demand, per frame | Learned features | Peer-aware, but demand-driven and per-frame; no last-transmitted state and nothing Bayes-fusable. |
| **Battistelli et al. 2019** | A Bayesian belief, on KL from a reference density | Node (whole state) | **Last transmission** | Full posterior | The mechanism, exactly, at node granularity on a shared state; we apply it per voxel to a conjugate counting posterior with an evidence arm the Gaussian case cannot express. |
| **Trimpe & Campi 2015** | Gaussian belief; KL trigger expanded into two terms | Node | Last transmission | Gaussian posterior | Their Eq. 5 is our decomposition. We instantiate the two terms on Beta and take a per-voxel deterministic bound instead of a scalar KL budget. |
| **Trimpe & D'Andrea 2014** | State estimate, on variance **growth** | Node | Absolute variance bound | Gaussian posterior | The dual: they fire when uncertainty grows under process noise; a static map has none, so we fire when it shrinks by a factor. |
| **Han et al. 2015** | Measurements, stochastic trigger | Message | Last transmission | Gaussian, closed form preserved | They avoid the censoring problem by randomising the trigger; we take a deterministic trigger and absorb the censoring, recovering exactness on each absolute send. |
| **Rocha et al. 2005** | Range measurements, by summed information utility | **Per measurement** (sum over affected voxels), Eq. 64/68 | **Sender's own map** | Raw measurements + pose, re-integrated by each receiver | Censoring-sensors, not send-on-delta: peer-memoryless, and ships measurements so no posterior is ever degraded. Their per-voxel term `log(σ/σ')` is our evidence arm's measure, used as a summand rather than as the threshold. |
| **Kimera-Multi** (Tian 2022) | Keyframes, loop-closure edges, mesh segments | Keyframe / graph edge | Own pose graph | Metric-semantic mesh; no occupancy counts | Sparsifies the *graph* layer; the dense volumetric layer is untouched — which is the layer we address. |
| **Swarm-SLAM** (Lajoie 2024) | Inter-robot loop-closure candidates, prioritised | Pose-graph edge | Own graph + priority heuristic | Sparse pose graph | Bandwidth saved on localisation, not mapping. Orthogonal and composable, not competing. |
| **Reijgwart et al. 2023** | Wavelet coefficients, thresholded | Multi-scale coefficient | Own map | Lossily reconstructed occupancy | Compresses the representation and skips saturated regions; we skip *transmissions* and leave the representation exact. |
| **Olston et al. 2001/2003** | Cached scalar values under precision bounds | Object / cached value | Last transmitted | Value within a guaranteed bound | The adaptive-τ contribution we explicitly do **not** claim; our fixed global τ is their stated naive baseline. |
| **US 12,236,779 B2** (Intel) | Layered cost-map data | **Layer** | Confidence sufficiency | Cost-map layers | Layer granularity, not cell; different level of the hierarchy. |

---

## 3. Related-work outline

Six paragraphs. The order is load-bearing: it walks a reviewer from "why the
substrate creates the problem" to "why nobody's answer applies" before the
contribution appears.

**¶1 — The conjugate volumetric substrate and the traffic it generates.**
*Claim:* Beta/Dirichlet pseudo-count voxels retain the uncertainty that log-odds
collapse, and that retention is precisely what makes them expensive to share —
continuous ray-carving increments produce a stream of genuinely-changed but
operationally-identical values. Cite Doherty 2017, Gan 2020, Hornung 2013 (for
the log-odds contrast and the clamping policy). Ground the claim in the measured
7× re-send factor over 5.56 M coords.

**¶2 — Bandwidth in collaborative SLAM, and the layer nobody addresses.**
*Claim:* Distributed SLAM systems have optimised communication hard, but at the
pose-graph and keyframe layer; the dense volumetric layer is either not shared or
shared naively. Cite Kimera-Multi, DOOR-SLAM, Swarm-SLAM, DiSCo-SLAM, and the
2024–2026 distillation line (Hermes et al.) once verified. Position as
**orthogonal**, not competing — this is the paragraph that stops a reviewer
demanding an ATE comparison.

**¶3 — Deployed volumetric map sharing gates on discrete state, and pays for it
with the posterior.** *Claim:* every deployed low-bandwidth volumetric sharing
system reduces bytes by degrading what the peer holds. Cite OctoMap change
detection, MARBLE (binary payload), `vdb_mapping` (with the README concession),
Łuczyński 2017 (with the §VI.A quotes — the prior art stating the pathology
itself), Reijgwart 2023. **This is the paper's central paragraph.** Everything
before it is setup.

**¶4 — Learned compression buys orders of magnitude and forfeits fusability.**
*Claim:* feature-level collaborative perception achieves compression this method
cannot approach, and produces representations that cannot be Bayes-fused,
audited, or consumed by a planner requiring explicit uncertainty. Cite
Where2comm, CoSDH, InfoCom (2026). Carve the niche explicitly and early;
per §7/O6 this is a scope argument, not a bandwidth argument, and it must not be
attempted on bandwidth numbers.

**¶5 — Event-triggered estimation: the mechanism, conceded.** *Claim:* gating
transmission on divergence from a last-transmitted reference is a mature idea
with priority in control, sensor networks, databases and industrial standards;
the contribution is not the trigger. Cite Miśkowicz, Åström & Bernhardsson, Årzén
(liveness), Trimpe & Campi (the two-term decomposition), Trimpe & D'Andrea (the
dual), Battistelli 2019, Mitra 2020, Rago 1996, and the standards block in one
sentence. Then Rocha 2005 with the three-way separation. **Concede first and
plainly** — a reviewer who finds Battistelli before the paper cites it will
reject on framing alone.

**¶6 — What the counting substrate changes.** *Claim:* additive pseudo-counts
make a suppressed update *subsumable* by the next absolute send, so the gate is
lossy in convergence rate rather than in value — the property no compression-based
or feature-based method has. Then the evidence arm as the arm only this substrate
can express. Close with the honest caveats: censored-Beta (Han 2015), the
fusion architecture and its preconditions (T2 — replica layers + refold; cite
Julier & Uhlmann and GCI as the treatment for incremental fusion, which this
system does not do), and adaptive τ as future work (Olston).

---

## 4. Threat register

Most severe first. Both source reviews' registers are merged and re-ranked; two
items are downgraded on the strength of the verification pass, and one new item
is added. T2 keeps its number but was closed post-review by a code read (§0.5).

### T1 — US 2023/0110467 A1 remains unread `[SEVERITY: HIGH]`

The only known disclosure combining **cell granularity** with a **threshold versus
previously transmitted messages**. If it reads as believed, it is mechanism prior
art at the same abstraction level as the gate. It is believed abandoned, which
makes it prior art rather than an enforceable claim — that reduces legal exposure
to zero and leaves novelty exposure untouched.

*To rebut:* pull the primary claim text and file wrapper. Chart the claims against
the mechanism on three axes: cell vs. layer, last-transmitted vs. last-updated,
and — the one that most likely separates them — whether the gated quantity is a
scalar cost/confidence or a conjugate posterior parameter pair. Also search
JP/CN/KR/DE, still untouched.

*Recommendation:* do not assert mechanism-level novelty in the abstract or intro
until this is read. The framing already concedes the mechanism, so a bad outcome
here costs a sentence, not the paper — provided the concession is written first.

### T2 — Data incest `[CLOSED — resolved 2026-07-31 by code read]`

Additive pseudo-counts merged across robots with shared observational ancestry
double-count evidence and produce overconfidence. Every reviewer with a
decentralized-fusion background will raise it — and the architecture answers it.

The ambiguity that previously blocked this (snapshot-replace at
[dscovox_node.cpp:462](../../src/scovox_mapping/src/dscovox_node.cpp#L462) vs.
`mergeBeta`'s prior-subtraction) dissolved on reading the code: both are real,
at different layers. The merger keeps one replica grid pair *per source robot*
(`sources_`, [dscovox_node.cpp:416](../../src/scovox_mapping/src/dscovox_node.cpp#L416));
snapshot-replace only ever overwrites a robot's *own* previous value inside that
robot's layer, so no receiver evidence is discarded — the layer never held any.
Each touched fused cell is then rebuilt from scratch: reset to prior, fold every
source's *current* value via `mergeBeta`/`mergeDir` (`refoldBeta`,
[dscovox_consensus.hpp:163](../../src/scovox_mapping/include/scovox/dscovox_consensus.hpp#L163)).
Prior-subtraction never runs against an accumulating fused value, and the header
names the property: the result "depends only on the current *set* of source
values, never on how many times a snapshot was received." No carve posterior
ever ingests fused output (`scovox_node` never subscribes to `ScovoxMapBinary`;
`dscovox_node` publishes only `ScovoxMap`), so the topology is acyclic and no
evidence path returns to its origin.

The consequence is stronger than the scope argument the paper was going to make:
**fused = F(current source replicas)** — a pure function. The gate changes
*when* a replica attains a value, never what the fold produces from it. With
respect to fusion the gate is exact per cell at every instant, not merely in the
limit; "exact in the limit" remains the correct claim for the individual link
(the replica holds the last value *sent*, not the sender's current one).

Two conditions keep the claim honest, and both belong in the paper as stated
preconditions rather than hedges:

- **Topology.** The proof rests on carve posteriors never ingesting fused
  output. A future peer-to-peer variant where robots fold peer maps into their
  own carve grids reintroduces incest immediately — unless it too keeps
  per-source replica layers and refolds. Cite Julier & Uhlmann and the GCI line
  as the treatment for architectures that fuse incrementally instead — the
  alternative, not ours. Do **not** claim a Beta/Dirichlet GCI result exists —
  §1.7 could not confirm one.
- **Semantics.** `mergeDir` re-truncates the union of the sources' top-K slots
  to K_TOP at every fold; a class dumped to OTHER cannot climb back (fold order
  is pinned by sorted source keys, so the result is deterministic). The
  occupancy fold is exact; the semantic fold is where the substrate's central
  hypothesis — that K_TOP = 2 + OTHER retains approximately all the information
  needed for uncertainty, memory, and wire bandwidth — is stressed hardest.
  That is an experiment the full paper runs on purpose, not a leak to paper
  over.

Review-01's threat 4.2 (max-count "conservative fusion" rebuttal) answers an
architecture the system does not have; adopting it would concede incremental
additive fusion. Drop it.

*Recommendation:* state the architecture in one paragraph (replica layers,
reset-and-refold, acyclic topology) and make the orthogonality claim as a proved
property of it, with the two preconditions above stated plainly.

### T3 — "Peer lag" is Age of Incorrect Information `[SEVERITY: MODERATE — cheap fix]`

The proposed metric — time until the receiver's value comes within ε of the
sender's — is AoII (Maatouk et al. 2020), which penalises staleness only when the
receiver is actually wrong. Both source reviews reached this independently and
both are right.

*Recommendation:* **adopt the name.** Call it AoII, or "belief AoII" if the ε-ball
on the Beta parameters needs distinguishing from AoII's discrete state mismatch.
Cite Maatouk 2020 and the Yates AoI survey. This converts a novelty liability
into cross-disciplinary rigour at the cost of a rename, and it is the single
highest-return edit in this document.

### T4 — Censored Beta: the exactness claim is over-stated as written `[SEVERITY: MODERATE]`

Deterministic deadbands break conjugacy. Between sends, the receiver's true
posterior is a censored Beta — the silence itself is information ("the sender has
not moved by more than τ") that the receiver is not conditioning on. Han et al.
2015 is the Gaussian precedent, and the reason stochastic triggers exist at all.
No treatment of the censored **counting-family** case was found; if that holds up
under a dedicated search, it is a genuine future-work paragraph rather than a
correctness obligation.

The design doc's own resolution is the right one and is stronger than either
source review's: because the wire carries **absolute** values, each transmission
overwrites the distorted state and restores exactness; and the bias direction is
conservative — the receiver's evidence is never more than κ× stale, so its
variance is at most ~κ× the sender's and the peer is *less* confident than truth,
never more. For a planner consuming the merged map, that is the safe direction,
and saying so explicitly is worth a sentence.

*Recommendation:* keep "exact in the limit" but bind it to the transmission
instant, and state the between-send state as a bounded, conservatively-biased
approximation. Cite Han 2015 while conceding. Do not let the phrase stand
unqualified in the abstract.

### T5 — 2024–2026 collaborative SLAM baselines are unverified `[SEVERITY: MODERATE]`

Hermes, Alice-SLAM and CCMD-SLAM reportedly demonstrate large bandwidth
reductions with preserved accuracy. All three arrive second-hand through
review-02 with no author lists and, for CCMD-SLAM, an incoherent venue. They
cannot be cited or compared against in this state, and one of them may be the
closest recent work in the document.

*To rebut:* retrieve the records. Then determine, per system, whether the receiver
holds a fusable posterior — if none does, they reinforce ¶3 rather than
threatening it, and the bandwidth-number comparison is a category error worth
making explicit.

### T6 — Evaluation scale `[SEVERITY: MODERATE — and it is the one that actually gets papers rejected]`

The measurement is a single 154-frame, single-robot replay. The target venues
expect ≥3 robots and real-robot results. A single-robot capture cannot exhibit
the effect the paper is about, because there is no peer.

*Recommendation:* treat this as the binding constraint on the submission
timeline, not as a rebuttal exercise. See §5 and §6.

### T7 — Goal-oriented / semantic communication may subsume the gate `[SEVERITY: LOW]`

The 2022–2026 semantic-communication line formalises "transmit only what changes
the receiver's decision." Review-02 checked and found the work model-based and
task-specific, with no general theorem covering thresholded transmission of
counting-family posteriors. Consistent with §1.6.

*Recommendation:* position the gate as a concrete analytical instantiation within
that framing rather than a new principle. Low cost, and it inoculates against a
communications-theory reviewer.

### T8 — Underwater acoustic map transmission is still unswept `[SEVERITY: LOW — but genuinely open]`

Downgraded from the brief's #1. Łuczyński turned out to be satellite
teleoperation (§0.3), so resolving it did not sweep the acoustic subfield — it
removed the only entry anyone had. Acoustic links (~1–10 kbps typical, not the
~100 kbps assumed) are the most bandwidth-hostile regime in robotics and remain
the most plausible location for an undiscovered aggressive per-cell gate.

*Recommendation:* one targeted sweep of OCEANS / IEEE JOE / AUV proceedings on
acoustic map transmission. If it comes back empty, say so in the paper — "we
found no per-cell continuous gate in the acoustic literature" is a defensible
sentence once the search is actually run, and an indefensible one before.

### T9 — The evidence arm's operational justification is asserted, not shown `[SEVERITY: LOW — NEW]`

Neither source review raised this. The argument for the evidence arm is that a
saturated voxel starved by a pure Δp gate leaves the peer unable to distinguish
"probably a wall" from "certainly a wall," causing an exploring robot to
redundantly revisit. That is a claim about a *downstream consumer*, and nothing in
the measurement or validation plan currently demonstrates it. The design doc's own
sanity bound sharpens the problem: past ~5.1 doublings the arm is transmitting
information OctoMap's default clamping deliberately discards.

*To rebut:* the planner-level metric already in the validation plan (step 3) has
to be the one that shows it — specifically, redundant revisit rate under
mean-arm-only versus mean+evidence. Without that number, Objection O3 in §7 lands.

*Recommendation:* promote this from "a metric we also collect" to the experiment
that justifies the secondary claim.

---

## 5. Baselines and evaluation precedent

### 5.1 Required baselines

The design doc's three are correct and sufficient; ordered here by what each
isolates.

1. **Current any-change gate** — the shipped behaviour. Establishes the 7×
   re-send figure. **Not the headline comparison**: the design doc is right that
   this baseline is weaker than stock OctoMap, and claiming a large win over it
   is the kind of thing a reviewer notices and punishes. State the caveat in the
   paper, not just in the design doc.
2. **State-flip gate (OctoMap-equivalent)** — the honest baseline. Public
   implementation: `octomap`, `use_change_detection` /
   `OccupancyOcTreeBase::updateNodeRecurs`. This is what the deployed literature
   would ask for, and Łuczyński independently implements the same rule, which is
   worth a sentence.
3. **State-flip + binarized payload (MARBLE-equivalent)** — **the one to beat, on
   map quality at equal bandwidth.** Public implementation:
   `github.com/dan-riley/marble_mapping`. If the gate does not win here the paper
   has no result.
4. **Pure Δp gate (mean arm only)** — the ablation that justifies the evidence
   arm, per T9.

Periodic/time-triggered transmission (review-01's suggestion) is worth including
as a trivial upper bound but isolates nothing; one line in a table, not a curve.

Learned compression baselines (Where2comm et al.) should be **excluded with a
stated reason**, not omitted silently — see §7/O6.

### 5.2 Metrics

- **Bandwidth** — bytes/s per robot and total per mission. Universal in this
  literature.
- **AoII** (not "peer lag") — integral of a penalty over intervals where the
  receiver's belief is outside the ε-ball. Cite Maatouk 2020. **The metric the
  binarizing systems structurally cannot measure**, which is the argument for
  reporting it prominently.
- **Posterior divergence** — KL between the receiver's map and the ungated
  centralized oracle. Both source reviews independently insist on KL over IoU
  here, correctly: IoU measures the classification the baselines preserve, and is
  blind to the confidence they destroy. Reporting IoU *as well* is worth doing
  precisely because it should show near-parity — that is the point.
- **Planner-level metric** — map quality only matters through consumers, and per
  T9 this is where the evidence arm is justified or abandoned.

### 5.3 The expected curve

Pareto frontier, bandwidth (x) against posterior error (y), one curve per gate
family, swept over τ ∈ {0.01, 0.02, 0.05, 0.1} × κ ∈ {1.5, 2, 4}. Reviewers
expect a visible knee and expect the proposed method to dominate toward the
origin. The MARBLE-equivalent baseline will appear as a **single point**, not a
curve — it has no tunable quality knob. Making that visual point explicit is
worth more than any individual number in the plot: the contribution is the
existence of the axis, not the position on it.

---

## 6. Venue reading and reviewer expectations

Positioning targets, in the order they should be read:

1. **Kimera-Multi** (T-RO 2022) — the reference point for distributed
   metric-semantic mapping and communication budgets.
2. **Swarm-SLAM** (RA-L 2024) — current sparse-communication CSLAM;
   loop-closure prioritisation.
3. **DiSCo-SLAM** (RA-L 2022) — descriptor-level sharing.
4. **Reijgwart et al.** (RSS 2023) — the volumetric-compression counterpoint, and
   the closest thing to a "skip the settled regions" argument in mapping.
5. **Where2comm** (NeurIPS 2022) and **CoSDH** (CVPR 2025) — the learned line the
   introduction must intercept.
6. **Hermes / Alice-SLAM** (RA-L 2025) — once verified (T5), likely the most
   recent direct competition.

**What these reviewers demonstrably expect**, on which both source reviews agree
and the venue record supports:

- **Fleet size ≥ 3.** Two robots no longer reads as multi-robot, and network
  congestion effects are non-linear in fleet size — a two-robot result cannot
  speak to them.
- **Real robots, or a recognised public dataset.** Simulation-only is rejected
  absent a major theoretical contribution. If hardware is unavailable, use S3E or
  the Kimera-Multi datasets rather than a private replay.
- **Ablation depth.** Per-arm isolation (mean; mean+evidence; all three), plus
  sensitivity on τ and κ. Non-negotiable for a three-arm mechanism.
- **Robustness to packet loss and variable bandwidth.** Specifically relevant
  here because the liveness arm exists *because* suppression and loss are
  indistinguishable to the receiver; a reviewer who spots that will ask for the
  lossy-link experiment.

The gap between this and the current single-robot 154-frame capture (T6) is the
main obstacle to submission, and it is an experimental gap rather than a
literature gap. Nothing in this document reduces it.

---

## 7. What a hostile reviewer says

**O1 — "This is event-triggered estimation with a new coat of paint."**
Battistelli 2019 gates a Bayesian belief on KL from the last transmission. Trimpe
& Campi decompose exactly this trigger into exactly these two terms.
*Answer — concede, then narrow.* The mechanism is conceded in ¶5 before the
reviewer reaches it. What is new is per-voxel application to a volumetric map at
5.5 M-coordinate scale, on a conjugate counting posterior whose evidence arm has
no Gaussian analogue (Trimpe & D'Andrea's variance trigger is its *dual*, and
does not cover the shrinking-variance regime), plus the empirical characterisation.
Never claim "KL-triggered gating" as such.

**O2 — "'Exact in the limit' is false. Deadbands censor the posterior."**
*Answer — concede and bound.* Correct as stated for the between-send interval;
the receiver holds a censored Beta and is not conditioning on the silence. The
exactness claim attaches to the transmission instant, and holds only because the
wire carries absolute pseudo-counts — the distinguishing property versus delta
coding, which the design doc rejected for independent architectural reasons. Add:
the bias is conservative (peer under-confident, never over-confident, bounded by
κ). Cite Han 2015. Fix the abstract wording; do not defend the unqualified phrase.

**O3 — "The evidence arm is a solution in search of a problem."**
A planner treats p = 0.9 and p = 0.99 identically; spending bandwidth to upgrade
"certain wall" to "very certain wall" is exactly what goal-oriented communication
says not to do.
*Answer — currently incomplete (T9).* The argument is that entropy-driven
exploration, unlike collision checking, is sensitive to the evidence count, so a
starved voxel provokes redundant revisits. **This must be shown, not asserted** —
redundant revisit rate under mean-arm-only versus mean+evidence. Also concede the
bound: past OctoMap's clamping equivalent (~5.1 doublings) the arm is shipping
information a conventional map discards by design. Without the planner number,
this objection lands.

**O4 — "`vdb_mapping` already ships an overwrite grid. Gating a float instead of a
bool is an engineering delta."**
*Answer — the delta is the entire point, and their README says so.* FZI's own
documentation concedes the mode "has the downside, that the probabilistic
framework is lost on the remote side." The contribution is not the overwrite grid;
it is achieving comparable bandwidth *without* that downside. Łuczyński (§VI.A)
independently confirms the pattern: state-flip gate on the wire, log-odds retained
on-vehicle. Two deployed systems documenting the same loss is the strongest
evidence the paper has that the loss is not incidental.

**O5 — "Additive counts across robots double-count. The gate is irrelevant if the
fusion is wrong."**
*Answer — the fusion never adds across history (T2, resolved).* The merger keeps
a replica layer per source and rebuilds each fused cell from scratch — reset to
prior, fold the sources' current values — so the fused cell is a pure function of
the current replicas, and no carve posterior ever ingests fused output. There is
no accumulation for the gate to corrupt: gating changes *when* a replica attains
a value, not what the fold produces from it. State the architecture in one
paragraph; cite Julier & Uhlmann and GCI as the treatment for systems that do
fuse incrementally, and do not claim a Beta/Dirichlet GCI result exists.

**O6 — [EVALUATION] "You omitted the only baselines that matter. Learned
compression beats this by orders of magnitude."**
*Answer — concede the metric, contest its relevance; do not fight on bandwidth.*
InfoCom-class methods reach <10 KB per collaboration; pseudo-count transmission
cannot approach that and should not pretend to. The niche must be carved in the
**introduction**, not the rebuttal: systems requiring an interpretable posterior
for formal safety arguments, for auditability, or for CPU-only deployment cannot
consume learned features regardless of their size. Exclude the baselines with the
reason stated in the text. An unstated omission here is fatal; a stated one is a
scope decision.

**O7 — [EVALUATION] "One 154-frame single-robot replay."**
*Answer — no answer. Must be fixed before submission (T6).* A single-robot capture
cannot exhibit the phenomenon the paper is about — there is no peer, so peer lag,
AoII and merged-map divergence are all undefined. This needs ≥3 agents on a public
dataset or hardware, plus a lossy-link condition.

**O8 — [FRAMING] "Your motivating measurement is against your own weak baseline."**
The 7× re-send figure is measured against an "emit if anything changed" gate that
sends strictly more than stock OctoMap, whose change detection suppresses exactly
the ray-carving traffic being counted.
*Answer — concede loudly and pre-emptively.* The design doc already identifies
this and it is to the paper's credit; the honest framing is that the state-flip
gate is a **degenerate significance gate** with threshold "crossed 0.5," and the
proposal is its graded generalisation. Report the headline number against the
state-flip baseline, and present the 7× figure only as characterisation of the
current implementation. Reporting it as the headline would be the single most
damaging framing choice available.

**O9 — [FRAMING] "'Peer lag' is AoII."**
*Answer — rename (T3).* Adopt AoII and cite Maatouk 2020. Free.

**O10 — "Why not compute the Beta KL directly, since it is closed-form and cheap?"**
*Answer — deliberate, and defensible on grounds other than cost.* The design doc
is right that "KL is expensive" is not a defensible justification and must not
appear. The real reason is that the decoupled two-arm form provides a hard
per-voxel deterministic guarantee — `|p_recv − p_send| ≤ τ` everywhere, always —
that a scalar KL budget does not. For a map consumed by a planner, a per-voxel
bound is worth more than an aggregate information budget. Concede the cost: the
two arms are in mismatched units (the evidence arm is constant-KL at ½ ln 2 per
doubling; the mean arm's KL grows with n at fixed τ), so a fixed τ over-transmits
when evidence is thin and under-transmits when it is thick. τ ∝ 1/n is the
correction and is already in the ablation plan — put it there before a reviewer
asks.

---

## Appendix A — Citation audit

Errors found in the two source reviews. Recorded so they are not reintroduced.

**A.1 — AoII DOI.** Review-01 gives `10.1109/TNET.2020.2990202`. Correct:
`10.1109/TNET.2020.3005549`, IEEE/ACM ToN 28(5):2215–2228, 2020, Maatouk,
Kriouile, Assaad, Ephremides. Review-02 gives the year as 2019 and the venue as
"IEEE TIFS / IEEE JSAC" — also wrong; 2019 is the arXiv preprint (1907.06604).

**A.2 — DSP-Map.** Review-01: "Chen, Jing et al., *DSP-Map: Dual-Structure
Particle-Based Continuous Dynamic Occupancy Map*, T-RO 2023." Correct: Chen, G.,
Dong, W., Peng, P., Alonso-Mora, J., Zhu, X., "Continuous Occupancy Mapping in
Dynamic Environments Using Particles," *IEEE T-RO* 40:64–84, **2024**. Wrong first
author, wrong title, wrong year.

**A.3 — Han et al. 2015.** Review-01 pairs the title "Multi-sensor scheduling for
state estimation with event-based stochastic triggers" with DOI
`10.1109/TAC.2015.2505066`. These are two different papers. The censored-posterior
citation the argument needs is "Stochastic Event-Triggered Sensor Schedule for
Remote State Estimation," *IEEE TAC* 60(10):2661–2675, 2015, DOI
`10.1109/TAC.2015.2406975` (arXiv:1402.0599). The multi-sensor paper is
arXiv:1502.03068. The design doc had this right.

**A.4 — Collaborative SLAM attributions (all review-02).** Kimera-Multi is Tian
et al., **T-RO 2022**, not "Rosinol et al., RA-L 2021" (Rosinol is single-robot
Kimera). Swarm-SLAM is **Lajoie & Beltrame, RA-L 2024**, not Cieslewski.
DiSCo-SLAM is **Huang, Shan, Chen & Englot**, not "Wang, L." Three
first-author errors in one cluster; treat the rest of that section as suspect
until each record is retrieved.

**A.5 — "InfoCom" (review-01).** Given as "Zhang, Xinyu et al., *InfoCom:
Information-Aware Compression for Communication-Efficient Collaborative
Perception*, AAAI 2024," with an `ojs.aaai.org` URL. The real paper is "InfoCom:
Kilobyte-Scale Communication-Efficient Collaborative Perception with Information
Bottleneck," **AAAI 2026**. Title, venue-year, URL and authors are all wrong, and
the entry was tagged `[VERIFIED]`. **This is the entry that most warrants
distrusting review-01's verification tags generally.**

**A.6 — CoSDH (review-01).** First author given as "Xu, Derrick." Correct: Junhao
Xu, with Yanan Zhang, Zhi Cai, Di Huang. CVPR 2025 pp. 6834–6843,
arXiv:2503.03430. Paper is real; attribution is not.

**A.7 — GCI for Beta distributions (review-01).** Cited as Gao et al., *IEEE SPL*
2022, `10.1109/LSP.2022.3150242`, titled "Distributed GGIW-CPHD-based extended
target tracking over a sensor network," with the role "derives GCI specifically
for Beta-distributed detection probabilities." The title describes extended-target
tracking; it does not support the claimed role. **Do not cite for that claim.** No
Beta/Dirichlet-specific GCI result was confirmed to exist (§1.7).

**A.8 — Łuczyński et al. characterisation (all three documents).** Described as
~100 kbps *acoustic* links. It is a Ku-band **satellite** link, 768 kb/s up /
256 kb/s down, 620 ms RTT, single-vehicle ROV teleoperation, evaluated in
simulation. Review-01 further describes the gate as operating on "bounding boxes
of changed discrete states"; the bounding box selects resolution and colour for an
operator-specified ROI, while the gate itself is a per-voxel state-flip test
(§VI.A). See §0.2–0.3.

**A.9 — Where review-02 was right and review-01 was not.** Review-02 marked
Łuczyński `[UNVERIFIED]`, declined to characterise its gate, and explicitly listed
"obtain and read full text" as the rebuttal requirement. Review-01 asserted a
specific gate mechanism and tagged it `[VERIFIED]`. Review-02's epistemic caution
was correct and review-01's confidence was not — worth weighing when the two
sources conflict elsewhere in this document.

---

## Appendix B — Outstanding verification tasks

Ordered by value.

1. **US 2023/0110467 A1** — primary claim text and file wrapper (T1). Plus JP/CN/KR/DE.
2. **Hermes, Alice-SLAM, CCMD-SLAM** — retrieve records; determine what the
   receiver holds in each (T5).
3. **Acoustic map transmission sweep** — OCEANS / IEEE JOE / AUV proceedings (T8).
4. **Trimpe & Campi 2015, Eq. 5** — confirm the equation number and the two-term
   form against the PDF. The "principled decomposition" argument depends on it.
5. **`vdb_mapping` README quote** — confirm against current `main`, cite with a
   commit hash. It is the single most useful sentence in §3/¶3.
6. **Censored counting-family posteriors** — dedicated search. Determines whether
   T4 is a future-work paragraph or a correctness obligation.
7. **DOIs** for Doherty 2017, Gan 2020, and the design-doc-sourced control-theory
   entries in §1.2.
8. **Marck & Sijs** — resolve which paper (and which year) is meant (§1.2).
