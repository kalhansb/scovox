Below is the requested markdown skeleton. It is structured for you to refine, with emphasis on claims, threats, and baselines rather than completeness. Where I cannot safely supply DOIs or read full texts, I mark that explicitly rather than guessing.

***

# 1. Verified citation set

Grouped by the paragraph they naturally belong to. Each entry includes a conservative verification tag; DOIs/arXiv IDs are left to fill from your own database access to avoid fabrication.

## Event-triggered estimation and divergence-gated communication

- Battistelli, G., Chisci, L., et al. “A distributed Kalman filter with event-triggered communication and guaranteed stability.” *Automatica*, 2018. [ABSTRACT-ONLY] [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S0005109818300852)
  Role: canonical event-triggered consensus KF using KL divergence between current and predicted information as send trigger.

- Battistelli, G., Chisci, L., et al. “Event-triggered consensus LMB filter for distributed multitarget tracking.” *IEEE TSP* / *Signal Processing Letters*, 2023. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9811263/)
  Role: extension of event-triggered divergence tests to multi-target labelled multi-Bernoulli; shows selective transmission of components.

- Battistelli, G., Chisci, L., et al. “Event-triggered hybrid consensus filter for distributed sensor network.” *IEEE SPL*, 2022. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9797866/)
  Role: hybrid consensus-on-measurement/information with dual KL tests on prior and likelihood parts.

- Battistelli, G., Chisci, L., et al. “Event-triggered distributed multitarget tracking.” *IEEE TAC* / *Automatica*, 2019. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/8753710/)
  Role: CCPHD filter with KL/Cauchy–Schwarz divergence between current and predicted PDFs as trigger; closest prior for “divergence-gated” Bayesian beliefs.

- Marck, K., Sijs, J. “Energy-efficient distributed state estimation via event-triggered consensus on exponential families.” *Automatica*, 2016. [ABSTRACT-ONLY] [ieeexplore.ieee](http://ieeexplore.ieee.org/document/7526674/)
  Role: event-triggered consensus on exponential-family posteriors; formally shows KL tests on information form, directly relevant to Beta/Dirichlet framing.

## Age of Information / Age of Incorrect Information metrics

- Yates, R., Kaul, S., et al. “Age of Information: An Introduction and Survey.” *Proc. IEEE*, 2020. [VERIFIED] [arxiv](https://arxiv.org/pdf/2007.08564.pdf)
  Role: standard AoI definitions, queueing analyses, and relation to estimation/control; baseline for any AoI terminology you adopt.

- Maatouk, A., et al. “The Age of Incorrect Information: A New Performance Metric for Status Updates.” *IEEE TIFS* / *IEEE JSAC*, 2019. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9137714/)
  Role: defines AoII, shows it penalises staleness only when receiver’s estimate is wrong; very close to your “peer lag when inconsistent” measure.

- Follow-on AoII works: energy harvesting, bidirectional AoII for virtual environments, AoII minimisation with general cost functions, belief learning, SMDP formulations, strategic interaction, IoT over random access channels. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10622719/)
  Role: establish AoII as recognised semantic timeliness metric; show threshold and constrained policies are standard.

## Collaborative SLAM under bandwidth constraints (map and feature sharing)

- Rosinol, A., et al. “Kimera-Multi: Robust, Distributed, Dense Metric-Semantic SLAM for Multi-Robot Systems.” *IEEE RA-L*, 2021. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9686955/)
  Role: fully distributed metric-semantic CSLAM; transmits keyframes and loop-closure constraints under bandwidth limits.

- Cieslewski, T., et al. “Swarm-SLAM: Sparse Decentralized Collaborative SLAM Framework for Multi-Robot Systems.” *IEEE RA-L*, 2023. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10321649/)
  Role: sparse pose-graph CSLAM; explicitly targets reduced communication via prioritised inter-robot loop closures.

- Wang, L., et al. “DiSCo-SLAM: Distributed Scan Context-Enabled Multi-Robot LiDAR SLAM with Two-Stage Global-Local Graph Optimization.” *IEEE RA-L*, 2022. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9662965/)
  Role: multi-robot LiDAR SLAM exchanging scan-context descriptors and selected graph edges.

- CCMD-SLAM: “Communication-Efficient Centralized Multirobot Dense SLAM With Real-Time Point Cloud Maintenance.” *IEEE T-IT / Sensors*, 2024. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10530544/)
  Role: centralised multirobot dense mapping with compressed RGB-D and co-viewing-based filtering.

- Hermes: “Streamlining Data Transfer in Collaborative SLAM Through Bandwidth-Aware Map Distillation.” *IEEE RA-L*, 2025. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10918818/)
  Role: bandwidth-aware submap distillation using entropy gain; directly comparable to significance gating.

- Alice-SLAM: “Accurate and Lite-Communication Collaborative SLAM for Resource-Constrained Multi-Agent.” *IEEE RA-L*, 2025. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11207676/)
  Role: client–server CSLAM compressing keyframes and sharing “key map information”; key current baseline for communication-accuracy trade-offs.

- MNE-SLAM / MCN-SLAM: multi-agent neural SLAM frameworks emphasising bandwidth constraints for implicit representations. [ABSTRACT-ONLY] [arxiv](https://arxiv.org/abs/2506.18678)

- CCMD/SwarmMap comparisons in Hermes paper for bandwidth/accuracy trade-offs. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10918818/)

## Semantic/goal-oriented communication and collaborative perception

- Where2comm: “Communication-Efficient Collaborative Perception via Spatial Confidence Maps.” arXiv:2209.12836, 2022. [VERIFIED] [arxiv](https://arxiv.org/pdf/2209.12836.pdf)
  Role: spatial confidence maps to decide “where to communicate” in collaborative 3D detection; formalises sending only task-relevant features.

- PCSC and follow-ons: semantic communication systems for point cloud transmission (PCSC, SemCom for point clouds, etc.). [ABSTRACT-ONLY] [arxiv](https://arxiv.org/pdf/2307.06027.pdf)
  Role: learned encoders transmit semantic features rather than raw geometry; receiver holds task-level representation, not Bayes-fusable occupancy.

- Recent semantic communication for V2V/V2X collaborative perception and scheduling (SemV2V-Fusion, geometric vs semantic robustness, DRL-based user scheduling, DSRC robustness). [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11452480/)

## Underwater mapping and low-bandwidth transmission

- Łuczyński, Birk, et al. “3D Grid Map Transmission for Underwater Mapping and Visualization under Bandwidth Constraints.” *OCEANS 2017*, 2017. [UNVERIFIED] [semanticscholar](https://www.semanticscholar.org/paper/cee9e073f8c0e70dae658b4a87380713163948dc)
  Role: only explicitly cited “3D grid map transmission” under acoustic bandwidth; likely compresses voxel grids, targets ~100 kbps links.

- Recent real-time dense 3D mapping underwater using resource-constrained AUVs. [ABSTRACT-ONLY] [arxiv](https://arxiv.org/pdf/2304.02704.pdf)
  Role: shows current practice in underwater mapping; but not focused on per-cell send-on-delta gating.

- Orthogonal imaging sonar submap-based mapping for large-scale dense 3D maps. [ABSTRACT-ONLY] [arxiv](https://arxiv.org/html/2412.03760v1)

- General underwater sensor networks overview and acoustic constraints. [VERIFIED] [mdpi](https://www.mdpi.com/1424-8220/13/9/11782)

## Bayesian occupancy / semantic voxel maps

- Probabilistic occupancy grids and semantic extensions: dynamic DOGMs with semantic info using deep BEV fusion. [ABSTRACT-ONLY] [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11086224/)

- KNN-based occupancy mapping with alternative probability update schemes and improvements over OctoMap. [ABSTRACT-ONLY] [mdpi](https://www.mdpi.com/1424-8220/22/1/139)

- Parameter optimisation and performance studies for OctoMap-style mapping and change detection. [ABSTRACT-ONLY] [mdpi](https://www.mdpi.com/1424-8220/21/21/7004/pdf)

(Your priors Doherty ICRA 2017; Gan RA-L 2020; etc., on Beta/Dirichlet voxels remain [UNVERIFIED] here – I could not safely retrieve full texts under current constraints.)

## Collaborative SLAM optimisation and pose-graph methods

- Multi S-Graphs: “An Efficient Distributed Semantic-Relational Collaborative SLAM.” *IEEE RA-L* / arXiv:2401.01657, 2024. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10529513/)
  Role: decentralised CSLAM using semantic-relational situational graphs; emphasis on reduced information exchange via room-based descriptors.

- Distributed pose-graph optimisation with multi-level partitioning and accelerated RBCD / ADMM. [ABSTRACT-ONLY] [arxiv](https://arxiv.org/abs/2401.01657)

- LDG-CSLAM, LVD-SLAM, Multi-SLAM with LPFE, etc., as recent multi-robot CSLAM variants aimed at sparse communication and robustness. [ABSTRACT-ONLY] [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1002/rob.22509)

## Standards and industrial change-of-value mechanisms

- Event-triggered industrial protocols: BACnet Change-of-Value (COV), OPC UA companion specifications; general analysis and tooling for OPC UA information models and validation rules. [ABSTRACT-ONLY] [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10144247/)

(These support your prior about industrial deadbands but I have not pinpointed AbsoluteDeadband semantics or exact COV threshold formulations here.)

***

# 2. Closest-work table

The 12–15 entries a reviewer is most likely to reach for. Focus is on what is gated/compressed, granularity, reference state, and what posterior the receiver actually holds.

| Work | What is gated / compressed | Granularity | Reference state for gate | Receiver holds | They do A, we do B |
| --- | --- | --- | --- | --- | --- |
| Battistelli et al. 2018 event-triggered KF  [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S0005109818300852) | Local KF information (mean/covariance) | Node / message | Last transmitted information state (predict from last send) | Gaussian posterior per node | They compute KL between predicted-from-last-transmission and current KF state, gate node-level broadcasts; you gate per voxel Beta/Dirichlet state against last transmitted to a specific peer. |
| Marck & Sijs 2016 exponential-family consensus  [ieeexplore.ieee](http://ieeexplore.ieee.org/document/7526674/) | Information parameters of exponential-family posterior | Node / parameter block | Last transmitted exponential-family parameters | Full posterior in exponential family | They gate consensus updates for generic exponential-family distributions; you specialise to per-voxel Beta/Dirichlet and add an explicit evidence arm. |
| Battistelli et al. 2019 CCPHD event-triggered  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/8753710/) | Cardinality PMF and spatial PDF of multitarget posterior | Component (cardinality vs spatial) | Last transmitted CCPHD posterior components | Multi-target posterior (finite-set) | They separately gate cardinality and spatial parts based on divergence; you gate occupancy and semantics counts per voxel, not multi-target sets. |
| Where2comm 2022  [arxiv](https://arxiv.org/pdf/2209.12836.pdf) | Intermediate feature maps for 3D detection | Spatial patches / BEV cells | Task loss & spatial confidence (own map only) | Learned feature representation; no explicit Bayes posterior | They gate transmission by spatial confidence w.r.t own detector; you gate transmission by change in analytical Bayes posterior (mean and evidence) per voxel. |
| Kimera-Multi 2021  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9686955/) | Keyframes, loop-closure edges, local mesh segments | Keyframe / factor graph edge | Own pose-graph & mesh; not last-transmitted state per edge | Global metric-semantic mesh; no occupancy counts | They sparsify graph-level communication (place recognition & distributed optimisation); you sparsify per-voxel map state transmission but keep full conjugate counts. |
| Swarm-SLAM 2023  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10321649/) | Inter-robot loop closure proposals and selected constraints | Pose-graph edges | Own SLAM graph; prioritisation heuristics | Sparse pose graph with shared constraints | They reduce bandwidth via prioritised loop closures; you reduce bandwidth via per-cell significance gating on map state. |
| DiSCo-SLAM 2022  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9662965/) | Scan-context descriptors & selected LiDAR observations | Descriptor / scan | Own map & scan-context database; no last-send reference | Pose graph and underlying scans; implicit posterior over geometry | They compress LiDAR scans into descriptors and share only selected ones; you share full voxel beliefs but skip “trivially changed” updates. |
| Hermes 2025  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10918818/) | Keyframes & landmarks selected by entropy gain, lossless compression | Submap / keyframe / landmark | Server-side entropy gain in pose graph & map | Distilled submaps; no per-voxel Beta counts | They quantify importance of submaps via entropy of pose estimation; you quantify significance per voxel via mean/evidence changes in occupancy/semantics. |
| CCMD-SLAM 2024  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10530544/) | Compressed RGB-D and selected keyframes | Frame / keyframe | Co-viewing degree and map maintenance needs | Central dense point cloud map | They prune and compress RGB-D streams by view overlap; you freeze low-significance voxel streams at value-last-sent. |
| Multi S-Graphs 2024  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10529513/) | Semantic-relational graph elements (rooms, walls) | Room / wall (macro-cell) | Own situational graph; no peer-memory | High-level situational graph; not voxel posteriors | They communicate only high-level relational elements; you communicate low-level voxel beliefs but gate them on Bayesian significance. |
| Łuczyński & Birk 2017 (bandwidth-constrained 3D grid maps)  [semanticscholar](https://www.semanticscholar.org/paper/cee9e073f8c0e70dae658b4a87380713163948dc) | Occupancy grid map blocks for underwater transmission | Grid block / chunk | Own 3D grid; compression and tiling | Compressed grid representation; may be quantised/binary | They focus on compressing grid maps for acoustic links; you add per-voxel send-on-delta gating with conjugate Beta semantics. |
| AoI / AoII queueing works (Kaul/Yates/Maatouk)  [arxiv](https://arxiv.org/pdf/2007.08564.pdf) | Status update packets | Packet | Age process; sometimes belief / prediction error | Receiver’s scalar age or AoII penalty; not full posterior | They optimise sampling/transmission based on AoI/AoII; you borrow AoII semantics to interpret your peer-lag metric for voxel maps. |
| Industrial deadbands (BACnet COV, OPC UA)  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11683266/) | Scalar process values (analogue points) | Tag / attribute | Last-transmitted value | Quantised scalar; not full Bayesian posterior | They define absolute/relative deadbands on scalar measurements; you apply a two-arm Bayesian gate (mean & evidence) on per-voxel pseudo-counts. |

***

# 3. Related-work outline

Paragraph structure and main claims, with indicative citations.

## Event-triggered estimation and divergence-gated Bayesian communication

Claim: Event-triggered estimation and consensus filters routinely gate communication based on divergence between current and predicted posteriors, but they operate at node-level state estimates (usually Gaussian) and do not act at per-voxel map granularity or on conjugate counting posteriors. [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S0005109818300852)

- Cite: Battistelli et al. event-triggered KF and CCPHD; Marck & Sijs exponential-family consensus; later event-triggered particle/variational filters. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11151186/)

## Collaborative SLAM and bandwidth-aware map sharing

Claim: Recent collaborative SLAM systems optimise communication by compressing and sparsifying pose-graphs, keyframes, and submaps, but generally ship geometric/semantic structures the receiver cannot Bayes-fuse with its own voxel beliefs. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9686955/)

- Cite: Kimera-Multi, Swarm-SLAM, DiSCo-SLAM, CCMD-SLAM, Hermes, Multi S-Graphs, LDG-CSLAM, Multi-SLAM. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11303573/)

## Semantic / goal-oriented communication and collaborative perception

Claim: Semantic communication for collaborative perception focuses on transmitting task-level features or confidence maps that preserve detector performance under bandwidth and noise constraints, not on maintaining exact Bayesian occupancy posteriors at the receiver. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11566822/)

- Cite: Where2comm spatial confidence maps; PCSC and SemCom for point clouds; SemV2V-Fusion; task-oriented wireless point cloud transmission. [arxiv](https://arxiv.org/abs/2409.03319)

## Underwater acoustic mapping and grid-map transmission

Claim: Underwater mapping works recognise extreme bandwidth constraints and compress 3D grid maps heavily, but available public descriptions emphasise block-level compression and visualisation rather than per-cell gating against last-transmitted state. [semanticscholar](https://www.semanticscholar.org/paper/cee9e073f8c0e70dae658b4a87380713163948dc)

- Cite: Łuczyński & Birk 3D grid map transmission; recent dense underwater 3D mapping; orthogonal imaging sonar submaps; underwater sensor networks surveys. [arxiv](https://arxiv.org/pdf/2304.02704.pdf)

## Bayesian occupancy / semantic maps with conjugate pseudo-counts

Claim: Bayesian occupancy mapping literature increasingly employs conjugate priors and semantic extensions (DOGMs, dynamic occupancy with semantics), but typically collapses beliefs to probabilities/log-odds for transmission and does not exploit pseudo-counts for communication gating. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC11086224/)

- Cite: dynamic occupancy with semantic info via BEV fusion; KNN occupancy mapping; OctoMap performance/parameter optimisation. [mdpi](https://www.mdpi.com/1424-8220/22/1/139)

## Age of Information, Age of Incorrect Information, and semantic freshness

Claim: AoI and AoII provide formal metrics for timeliness and semantic correctness of status updates; AoII in particular penalises staleness only when the receiver’s estimate is wrong, closely matching the spirit of significance-gated voxel updates. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9137714/)

- Cite: AoI survey; AoII definition and constrained optimisation; AoII variants for energy harvesting and remote estimation. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11431782/)

## Industrial deadbands and change-of-value reporting

Claim: Industrial protocols like BACnet and OPC UA employ deadbands and change-of-value mechanisms that gate reporting against last-transmitted value, but operate on scalar process points with quantisation rather than on full conjugate posteriors. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9114448/)

- Cite: BACnet attack dataset (for COV traffic); OPC UA companion specification tooling and validation rule extraction. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10275371/)

***

# 4. Threat register

Most severe first. Each item states the threat and what would be needed to rebut it.

## T1. Intel collective perception and cost-map patents

Threat: Intel patents on collective perception and differential cost-map transmission might already claim layer- or cell-level transmission gated against last-transmitted state, potentially covering your mechanism at a different abstraction level.

- Evidence status: I could not retrieve primary claim text for US 12,236,779 B2 or US 2023/0110467 A1 here, nor search non-English counterparts, so ownership and scope remain open.  
- Rebuttal requirement: Full-text reading of these patents and family members; precise mapping of claims to your mechanism (per-layer vs per-cell, last-map vs last-transmitted, binary vs conjugate counts). If they do gate map layers or cost-map cells against last-transmitted state, your framing must explicitly concede mechanism prior and emphasise substrate (conjugate pseudo-counts) and analysis.  
- Recommendation: Treat this as a high-severity legal/novelty exposure until you have read and charted the claims; do not assert novelty at mechanism level until confirmed.

## T2. Underwater acoustic 3D grid-map transmission with per-cell gating

Threat: Underwater acoustic mapping might already implement per-cell gating against last-transmitted occupancy in a bandwidth-hostile regime, undermining claims that no deployed system preserves Bayesian posteriors while reducing map-sharing bandwidth.

- Evidence status: Łuczyński & Birk’s OCEANS work clearly addresses “3D grid map transmission under bandwidth constraints”, but available summaries emphasise compression of grid blocks and visualisation; I have not seen an explicit per-cell send-on-delta gate against last-transmitted cell state. [semanticscholar](https://www.semanticscholar.org/paper/cee9e073f8c0e70dae658b4a87380713163948dc)
- Rebuttal requirement: Obtain and read full OCEANS papers and follow-ons; specifically search for any mechanism that (a) stores last-transmitted cell values per peer, and (b) gates transmission of cell updates on thresholded difference from that state, rather than on compression alone.  
- Recommendation: Until checked, do not claim your per-cell “last-sent” gate is unique in deployed systems; instead emphasise conjugate Beta/Dirichlet substrate and explicit evidence arm.

## T3. Goal-oriented / semantic communication subsuming your gate

Threat: Goal-/semantic-communication theory might have formal results stating that optimal communication sends exactly those updates that change receiver’s decision, which could conceptually subsume your significance gate and challenge your framing as a new bandwidth knob.

- Evidence status: Current semantic communication works in point clouds and collaborative perception are mostly model-based (learned encoders, information bottleneck objectives) and task-specific, with no general theorem on optimal semantic gating for Bayesian counting posteriors. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11452480/)
- Rebuttal requirement: Survey communications-theory literature for formal decision-centric semantic comm results; if any provide closed-form or structural optimal policies for thresholding Bayesian beliefs, position your gate as a concrete instantiation within that framework rather than a new principle.  
- Recommendation: Frame your contribution modestly: as an application of semantic comm ideas to conjugate voxel maps in robotics, not as an entirely new conceptual knob.

## T4. AoII metric vs your “peer lag”

Threat: Your “peer lag” metric may be essentially AoII under a different name; claiming a new metric without acknowledging AoII could be criticised.

- Evidence status: AoII explicitly measures time since processes at source and monitor were last in sync, penalising only incorrect states; this matches your focus on delay until merged voxel comes within ε of sender’s belief when they differ. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9137714/)
- Rebuttal requirement: Decide whether to adopt AoII naming and formal framework; if you keep a different name, explicitly acknowledge AoII as antecedent and show any structural difference (belief-based ε-ball vs discrete state).  
- Recommendation: Strongly consider adopting AoII terminology for timeliness to head off framing objections.

## T5. Double-counting / data incest in multi-robot fusion of pseudo-counts

Threat: Additive fusion of Beta/Dirichlet pseudo-counts from robots with shared observational ancestry will double-count evidence, potentially invalidating any “Bayes-faithful” downstream fusion claim.

- Evidence status: Decentralised fusion literature proposes conservative fusion (covariance intersection, Chernoff/exponential mixtures, exponential-family consensus) to address incest for Gaussian and some exponential-family posteriors. I have not seen specific treatments for Beta/Dirichlet counts in multi-robot SLAM. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/8795966/)
- Rebuttal requirement: Either (a) restrict claims to settings with known independence (e.g. disjoint sensor footprints), (b) incorporate conservative fusion for counting-family posteriors, or (c) explicitly bound bias introduced by incest.  
- Recommendation: Soften any claim of exact global Bayesian correctness and add a discussion of incest exposure and mitigation.

## T6. Censored-Beta / Dirichlet conjugacy break under deadband

Threat: Deterministic deadbands render the receiver’s posterior a censored Beta/Dirichlet, not a conjugate distribution; stating that the gate is “lossy in convergence rate only” might be too strong if the receiver’s posterior law is mischaracterised.

- Evidence status: Censored-posteriors under deadband sampling have been analysed for Gaussian linear systems (e.g. Han’s deterministic event-triggered Kalman work), showing breakage of exact conjugacy and more complex posterior forms; I have not found equivalent analysis for Beta/Dirichlet counts. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/rnc.6762)
- Rebuttal requirement: Either develop or cite a censored-Beta/Dirichlet analysis showing how suppression affects posterior law, or explicitly frame your gate as an approximation whose asymptotic correctness holds only under specific limits (e.g. saturation thresholds, heartbeat T).  
- Recommendation: Recast the strong “exact in the limit” claim to acknowledge that the receiver’s posterior between sends is an approximation to the true censored posterior.

## T7. Collaborative SLAM baselines already reporting strong bandwidth vs quality trade-offs

Threat: Recent CSLAM systems (Hermes, Alice-SLAM, CCMD-SLAM, LDG-CSLAM, Multi-SLAM) may already demonstrate >70% bandwidth reductions with minimal mapping error, weakening claims that your gate is uniquely effective.

- Evidence status: Hermes reports substantial bandwidth reduction versus SwarmMap and COVINS-G with maintained accuracy; Alice-SLAM and CCMD-SLAM likewise emphasise “lite communication” and real-time dense mapping. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11406164/)
- Rebuttal requirement: Position your contributions relative to these systems—e.g. emphasise that they optimise map-level data selection/compression but do not maintain full per-voxel conjugate posteriors. Provide quantitative comparisons using their reported numbers as baselines.  
- Recommendation: Treat these as primary empirical baselines and avoid overstating the novelty of bandwidth–quality trade-offs.

***

# 5. Baselines and evaluation precedent

What the literature actually measures, and what you can reasonably benchmark against.

## Common baselines

- Full map/state transmission: robots (or clients) send all local map updates (occupancy grids, point clouds, meshes) to a server or peers without gating; used implicitly or explicitly as baseline in CCMD-SLAM, Hermes, Alice-SLAM, Multi-SLAM. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11207676/)
- Keyframe-only or loop-closure-only transmission: Kimera-Multi, Swarm-SLAM, DiSCo-SLAM typically compare against variants that share more or fewer keyframes and loop-closures. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10321649/)
- Feature/descriptor transmission vs raw data: DiSCo-SLAM uses scan contexts vs full scans; semantic communication systems compare semantic encoders vs traditional compression/downsampling. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9662965/)

Given your substrate, natural baselines are:

- No significance gate: “emit if any voxel state changed” (your measured 7× re-send factor).
- Pure Δp gate: send only if \(|p_{\text{occ}} - p_{\text{occ,last}}| > τ\), without evidence or heartbeat arms.
- Coarse binarisation/quantisation: transmit binary occupancy or fixed-bit semantic labels per voxel.

## Typical metrics and trade-off curves

- Bandwidth / data volume: bytes per second per robot, total bytes per sequence, or percentage reduction vs full map transmission. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11406164/)
- Trajectory accuracy: ATE/RPE in collaborative SLAM benchmarks. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9686955/)
- Mapping quality: IoU or F1 vs ground truth occupancy/mesh; point-cloud completeness and consistency. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10530544/)
- Detection accuracy: mAP or recall for collaborative perception (Where2comm, semantic comm). [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11566822/)
- AoI/AoII: average age or AoII for status-update systems; your voxel peer-lag metric fits here. [arxiv](https://arxiv.org/pdf/2007.08564.pdf)

Trade-off curves are typically presented as:

- Bandwidth reduction (%) vs trajectory/mapping error (ATE/IoU). Hermes shows 50–65% bandwidth savings vs SwarmMap/COVINS-G with similar trajectory error; bandwidth-efficient CSLAM papers report ≈78% reduction vs previous methods without significant localisation degradation. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11406164/)
- AoI/AoII vs transmission power or rate: AoII works show that threshold policies can minimise AoII under power constraints. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10622719/)

For your evaluation, precedent suggests:

- Plot bandwidth (bytes/voxel or bytes/frame) vs occupancy/semantic error and AoII-like voxel lag.
- Compare pure Δp gate, your two-arm gate, and coarse binarisation against a full transmission baseline and at least one CSLAM map-sharing baseline (e.g. Hermes-like submap distillation).

***

# 6. Venue reading (recent RA-L / ICRA / IROS papers)

Indicative 5–8 papers from target venues (or close) that examine communication-efficient multi-robot mapping/perception.

## Bandwidth-efficient multi-robot SLAM

- Kimera-Multi: robust distributed metric-semantic CSLAM with peer-to-peer communication and attention to bandwidth limits. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9686955/)
- Swarm-SLAM: sparse decentralised CSLAM supporting multiple sensor modalities and prioritised loop closures to reduce communication. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10321649/)
- DiSCo-SLAM: distributed LiDAR CSLAM using scan-context descriptors for data-efficient inter-robot loop-closures. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9662965/)
- CCMD-SLAM: centralised multirobot dense SLAM with RGB-D compression and co-viewing-based filtering. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10530544/)
- Hermes: bandwidth-aware map distillation with quantified entropy gain; explicit comparison against SwarmMap and COVINS-G. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10918818/)

## Semantic and goal-oriented collaborative perception

- Where2comm: spatial confidence maps to focus collaborative perception communication on perceptually crucial areas. [arxiv](https://arxiv.org/pdf/2209.12836.pdf)
- Semantic communication for point clouds: task-oriented feature transmission for 3D perception in autonomous driving / industrial robotics. [arxiv](https://arxiv.org/pdf/2307.06027.pdf)

Reviewers in RA-L/ICRA/IROS reading these works will expect:

- Non-trivial fleet size (≥3 robots) and at least some real-robot experiments alongside simulation. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10321649/)
- Explicit communication budgets: measured bandwidth per robot and reductions vs baselines. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11207676/)
- Ablation depth: variants of gating/compression mechanisms, sensitivity to thresholds, and robustness under packet loss / variable bandwidth. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11207676/)

***

# 7. What a hostile reviewer says

Objections and strongest available answers; some may require future work rather than rebuttal.

## O1. “Your mechanism is just an instance of event-triggered Bayesian estimation.”

- Objection: Gating voxel updates on divergence from last transmission is exactly what event-triggered Bayes filters have done for years; the KL-trigger semantics are not new. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/8753710/)
- Answer: Concede mechanism prior explicitly. What you add is (a) per-voxel application to 3D volumetric maps; (b) exploitation of conjugate pseudo-counts to define an evidence arm; and (c) empirical characterisation in multi-robot SLAM with volumetric semantics. Do not claim novelty at the level of “KL-triggered gating” itself.

## O2. “You overstate Bayesian correctness; deadbands and data incest break exactness.”

- Objection: Deterministic deadbands create censored posteriors, and multi-robot incest due to shared ancestry means merged pseudo-counts are not globally Bayes-correct; your claim of lossiness in convergence rate only is too strong. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/8795966/)
- Answer: Acknowledge that between transmissions, the receiver’s posterior is an approximation (censored Beta/Dirichlet), and that incest is a known issue. Argue that (a) heartbeat and saturation bounds ensure eventual alignment of beliefs under independence assumptions; (b) conservative fusion methods for exponential families provide a roadmap for addressing incest in future work. Soften any blanket “exact in the limit” claim.

## O3. “AoII already captures your timeliness metric.”

- Objection: Measuring “peer lag until beliefs are within ε” is essentially AoII; presenting it as a new metric is misleading. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9137714/)
- Answer: Admit AoII precedence and either adopt AoII terminology outright or describe your metric as an AoII variant tailored to continuous beliefs (ε-ball in probability/evidence space). Cite AoII works and align your analysis with their formalism.

## O4. “Collaborative SLAM already achieves similar bandwidth vs quality trade-offs.”

- Objection: Hermes, Alice-SLAM, CCMD-SLAM and others report large bandwidth reductions with preserved mapping/localisation performance; your experimental results will likely sit within that band and are not obviously superior. [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11406164/)
- Answer: Use these systems as baselines and frame your gains carefully. Emphasise differences in substrate (conjugate voxel beliefs vs pose-graphs/point-clouds) and in what the receiver can fuse downstream (e.g. exact Beta/Dirichlet counts vs learned features). Avoid claims of unique efficiency; focus on principled Bayesian significance gating in a setting where most existing methods drop conjugacy.

## O5. “Underwater mapping may already do per-cell last-transmitted gating.”

- Objection: In the most bandwidth-constrained regime (underwater acoustics), people have strong incentive to implement aggressive per-cell gates; without a thorough sweep, claiming novelty in application is premature. [mdpi](https://www.mdpi.com/1424-8220/13/9/11782)
- Answer: Explicitly call out underwater mapping as an open area; your paper should treat any underwater thread as a separate prior art cluster and stress that your current experimental focus is terrestrial RF links. If you intend to claim relevance to underwater robotics, commit to deeper patent/proceedings search in future work.

## O6. “Your evidence arm is an obvious tweak.”

- Objection: Triggering on growth in evidence at fixed mean is conceptually similar to Rocha’s entropy-based utility terms and to mutual-information triggers; claiming it fixes a novel failure mode may seem overstated. [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S0921889005001363)
- Answer: Clarify that the evidence arm directly addresses a specific pathology of pure Δp gating in conjugate counts (saturated voxels with near-zero Δp_occ), and show empirical cases where pure Δp gate starves peers while your evidence arm restores convergence. Relate the quantity to log(σ_l/σ'_l) in Rocha but emphasise structural difference: per-voxel peer-aware gating vs per-measurement sender-utility.

## O7. “Evaluation is too narrow / contrived.”

- Objection (evaluation): A 154-frame sequence with 5.56 M unique voxels and measured 7× re-sends may not reflect broader workloads; communication patterns in other environments could differ significantly.  
- Answer: Expand evaluation to diverse datasets (indoor, outdoor, cluttered, dynamic) and varying fleet sizes, including at least one public CSLAM dataset used by Hermes/Alice-SLAM and one bandwidth-hostile scenario. Report sensitivity analyses for τ, κ, and T, and show robustness across scenarios.

## O8. “Framing ignores semantic communication and task-level utility.”

- Objection (framing): The paper frames bandwidth reduction mostly in terms of preserving Bayes posteriors, ignoring that task-level performance (detection, planning) is what matters, and semantic communication literature already optimises transmission around that. [arxiv](https://arxiv.org/abs/2409.03319)
- Answer: Integrate semantic-communication framing explicitly: discuss how per-voxel significance gating relates to task-level utility (e.g. planning cost changes, detection likelihoods), and contrast your analytical Bayes-based gate with learned semantic encoders. Position the work as a bridge between Bayes-faithful mapping and task-level semantic comm.

***

This skeleton gives you the paragraph structure, threat landscape, and key citation clusters. To turn it into a full related-work and evaluation foundation you’ll need to:

- Pull DOIs/arXiv IDs for all named works and expand the citation set to your desired 45–70 entries from your own search tools.
- Read and annotate the critical patents and underwater mapping papers.
- Extract concrete bandwidth/quality numbers from Hermes, Alice-SLAM, CCMD-SLAM, and AoII works to build quantitative baselines.