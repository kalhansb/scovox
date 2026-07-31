# Literature-review prompt — significance-gated Bayesian map sharing

Paste everything below the rule into a fresh agent session (or hand it to a
subagent fleet). It is self-contained: it does not assume access to this repo.

Edit the **Parameters** block to retarget it. Everything after that is the brief.

---

## Parameters

- **Paper subject:** a per-voxel *significance gate* for multi-robot volumetric
  map sharing that reduces bandwidth without destroying the receiver's Bayesian
  posterior.
- **Target venues (in preference order):** IEEE RA-L, ICRA, IROS; fallbacks
  T-RO, Field Robotics, Autonomous Robots.
- **Stage:** pre-writing. No draft exists. I need the related-work foundation,
  the baseline set, and the evaluation precedent before I write a word.
- **Date of brief:** 2026-07-31. Treat anything after mid-2024 as the recency
  frontier that most needs sweeping.

---

# Brief

You are doing the literature review for a robotics systems paper. Your output
becomes the related-work section's skeleton, the experiment section's baseline
list, and the threat register I check before submission.

**This is not a novelty check.** A five-lens novelty survey has already run and
its findings are given below as established priors. Your job is to (a) turn
those findings into a citable, verified foundation, (b) close the gaps the
survey left open, and (c) find what it missed. Do not re-derive the priors —
but do flag any you believe are wrong, with evidence.

## 1. What the paper is about

**Substrate.** A fleet of ground/aerial robots each maintain a sparse voxel grid
of the environment. Every voxel carries a Bayesian belief in *conjugate
pseudo-count form*, not collapsed log-odds:

- occupancy: `Beta(a_occ, a_free)`, prior `Beta(1,1)`, so `p_occ = a_occ / (a_occ + a_free)`
  and total evidence `n = a_occ + a_free`; counts are capped at a saturation
  value (~1000) to retain adaptivity
- semantics: a `Dirichlet` over K classes, same counting structure

Robots stream **map state** — a delta of changed voxel records — to a merger
that combines fleet maps. Merge is snapshot-replace of the voxel record, so the
wire carries absolute values, not increments.

**The problem, measured.** Over a 154-frame capture: 5.56 M unique voxel
coordinates produced 38.87 M transmitted records (≈310 MB). Every voxel is
re-sent about **7 times**. Zero re-sends were byte-identical — the existing
"emit if anything changed" gate is already perfect at suppressing exact
duplicates. The traffic is dominated by *genuinely changed but trivially
different* values: ray carving nudges `a_free` upward by small increments,
moving `p_occ` by amounts a consumer cannot act on.

**The proposal.** Gate transmission per voxel against the value **last
transmitted to that peer**, with an OR of three arms:

```
send  iff  |p_occ − p_occ_last_sent| > τ                    (mean arm)
       or  (a_occ + a_free) > κ · (a_occ + a_free)_last_sent (evidence arm, κ ≈ 2)
       or  now − t_last_sent > T_heartbeat                   (liveness arm)
```

The two belief arms instantiate the two terms of a KL trigger on the voxel
belief — a mean-shift term and a precision-ratio term.

**The claimed contribution** (this is what the review must stress-test):

> Every deployed system that cuts map-sharing bandwidth does it by *degrading
> the posterior the peer receives* — binarizing occupancy, quantizing to a bit
> per leaf, or shipping learned features that cannot be Bayes-fused. Because
> the belief is held as additive pseudo-counts, a suppressed update is subsumed
> whole by the next absolute send. The gate is therefore lossy in **convergence
> rate only** and exact in the limit. It is a bandwidth knob that does not
> invalidate downstream Bayesian fusion.

Secondary claim: the **evidence arm** — firing on confidence growth at constant
probability — is only expressible because the substrate kept pseudo-counts, and
it fixes a real failure mode (a saturated voxel receiving confirming evidence
has Δ`p_occ` ≈ 0 and is starved forever by a pure Δp gate, so the peer never
learns the difference between "probably a wall" and "certainly a wall").

## 2. Established priors — verify, don't rebuild

The prior survey concluded: **the mechanism is not novel, the application is.**
Take these as given unless you find contrary evidence.

**Not novel; the paper must concede these explicitly.** Threshold-gated
"send-on-delta" transmission (Miśkowicz 2006; Åström & Bernhardsson 2002; OPC UA
`AbsoluteDeadband`; BACnet COV; DIS dead reckoning). Gating against the *last
transmitted* rather than last *updated* value (OctoMap `resetChangeDetection`).
Per-voxel gating of volumetric map transmission (OctoMap change detection, 2013;
deployed in MARBLE and FZI `vdb_mapping`). The Beta/Dirichlet voxel itself
(Doherty et al. ICRA 2017; Gan et al. RA-L 2020). Gating a Bayesian belief on
divergence from the last transmission (Battistelli et al. 2019; Marck & Sijs
2010).

**Nearest prior art, already resolved.** Rocha, Dias & Carvalho, *Robotics and
Autonomous Systems* 53(3–4):282–311, 2005 — entropy-based utility deciding what
map information is worth sending teammates. Distinguished on three grounds:
it thresholds a **per-measurement** sum of per-voxel information terms (Eq. 64,
Eq. 68) so no voxel is individually gated; it is **peer-memoryless** (utility is
measured against the sender's own map — "if the measurement is useful for itself
it is equally useful for its teammates"), making it the *censoring-sensors*
pattern rather than send-on-delta; and it ships **raw measurements plus pose**,
not map state, so posterior degradation cannot arise. Their per-voxel term
reduces to `log(σ_l/σ'_l)` — the same *quantity* as our evidence arm, differing
in structure, not in measure. Confirm this reading if you can obtain the paper;
challenge it if you disagree.

**Known open threats** (these are your highest-priority targets, §4).

## 3. Deliverables

Produce all six. Rank effort toward 1, 2 and 4.

1. **Verified citation set**, BibTeX-ready, grouped by the related-work
   paragraph it belongs to. Every entry needs a DOI or arXiv ID and a
   verification status (below). Target 45–70 entries; prune anything you cannot
   justify by its role in an argument.
2. **Closest-work table** — the 10–15 papers/systems a reviewer would name.
   Columns: work · what it gates or compresses · granularity (cell / message /
   layer / object) · reference state (none / own map / last-transmitted) ·
   what the receiver ends up holding (full posterior / binarized / learned
   features / raw measurements) · one sentence of "they do A, we do B".
   The reference-state and receiver-holds columns are where the contribution
   lives; be ruthless about filling them from the actual text, not the abstract.
3. **Related-work outline** — the paragraph structure, in order, with the
   citations assigned to each and a one-line claim per paragraph.
4. **Threat register** — anything that weakens or kills a claim, ranked by
   severity, each with what it would take to rebut. Say plainly if you think a
   claim should be dropped or softened. A single real hit here is worth more
   than twenty confirmatory citations.
5. **Baselines and evaluation precedent** — what this literature actually
   compares against and measures. Which baselines have public implementations.
   Which metrics recur, with the papers that established them, and what the
   bandwidth-vs-quality tradeoff curve conventionally looks like.
6. **Venue reading** — 5–8 papers from the target venues in the last three
   years that this must be positioned against, plus what those reviewers
   demonstrably expect (fleet size, real-robot vs simulation, ablation depth).

## 4. Specific gaps to close

Ordered by how much they could hurt.

1. **Underwater acoustic map transmission.** The most bandwidth-hostile setting
   in robotics (~100 kbps links) and the likeliest place an aggressive per-cell
   gate already exists. Start at Łuczyński & Birk (OCEANS 2017 and follow-ons)
   on 3-D grid map transmission; sweep the whole subfield, not just that thread.
   If someone there already gates per cell against last-transmitted state, the
   paper's framing changes.
2. **Patent re-verification.** Two filings were sourced second-hand through
   rate-limited services and must be checked against primary claim text:
   **US 12,236,779 B2** (Intel, collective perception service — believed to
   select at *layer* granularity on a fraction-of-cells-changed threshold) and
   **US 2023/0110467 A1** (Intel, believed abandoned — believed to disclose
   differential cost-map transmission of cells changed beyond a threshold versus
   previously transmitted messages). Report claim scope and legal status
   factually. Do **not** offer infringement or validity opinions. Also check for
   non-English filings (JP/CN/KR/DE), which were not searched at all.
3. **2024–2026 recency sweep.** The prior survey's coverage thins sharply after
   mid-2024. Collaborative perception has moved fast since Where2comm.
4. **Distributed / collaborative SLAM communication budgets.** Kimera-Multi,
   DOOR-SLAM, Swarm-SLAM, DiSCo-SLAM, maplab and successors. These report real
   bandwidth numbers and are what a robotics reviewer will benchmark the claim
   against. Determine what each actually transmits and at what granularity.
5. **Double-counting / data incest.** A reviewer will ask: additive
   pseudo-counts merged across robots that share observational ancestry
   double-count evidence. Find the decentralized-fusion literature on this
   (covariance intersection, Chernoff / exponential-mixture fusion, conservative
   fusion of Dirichlet and Beta beliefs) and tell me how bad the exposure is and
   who has addressed it for counting-family posteriors specifically.
6. **Age of Information.** The paper measures "peer lag" — time for a voxel's
   merged value to come within ε of the sender's. That is almost certainly a
   rediscovery. Check Age of Information (Kaul et al. 2012) and especially
   **Age of Incorrect Information** (Maatouk et al. 2020), which penalizes
   staleness only when the receiver's estimate is actually wrong — suspiciously
   close to this gate's semantics. Tell me whether to adopt the existing metric
   and its name outright.
7. **Goal-oriented / semantic communication.** The 2022–2026 communications-
   theory line on transmitting only what changes the receiver's decision. If it
   has a formal result that subsumes this gate, I need to know before a reviewer
   tells me.
8. **The censored-Beta problem.** Deterministic deadbands break conjugacy: after
   suppression the receiver's true posterior is a *censored* Beta, not a Beta
   (Han et al., TAC 2015, for the Gaussian case). Find whether anyone has worked
   the censored case for Beta/Dirichlet or any counting-family posterior. If
   nobody has, that is a future-work paragraph; if somebody has, it is a
   correctness obligation.

## 5. Search protocol

**The vocabulary problem is the whole difficulty.** This idea has a different
name in every field that reinvented it. Search each cluster on its own terms:

- *Control / estimation:* event-triggered estimation, event-based state
  estimation, send-on-delta, deadband sampling, Lebesgue sampling, level-crossing
  sampling, self-triggered, variance-based triggering
- *Detection / sensor networks:* censoring sensors, transmission control,
  suppression, value of information, sensor scheduling
- *Databases / streams:* adaptive filters, approximate caching, precision
  constraints, continuous queries, model-driven data acquisition
- *Robotics:* map sharing, map merging, communication-efficient collaborative
  perception, bandwidth-constrained multi-robot mapping, information-theoretic
  exploration, distributed occupancy mapping
- *Networking / standards:* change-of-value, exception reporting, dead
  reckoning, delta encoding, state synchronization, collective perception
  service, interest management (the games/DIS lineage)
- *Compression:* octree binarization, wavelet volumetric compression, submap
  sharing, learned map compression

Cross the mechanism terms with the substrate terms deliberately — "event-
triggered" × "occupancy grid", "deadband" × "voxel", "send-on-delta" × "map" —
because the fields do not cite each other.

**Sources.** If a scholarly-search tool is connected, use it first; it is
strictly better than web search for coverage and for resolving DOIs. Otherwise
web search plus direct fetch of arXiv, IEEE Xplore, ACM DL, ScienceDirect,
Semantic Scholar, DBLP. For deployed systems read the *source code and README*,
not the paper — the paper describes the algorithm, the code describes what
actually goes on the wire, and the gap between them is where this paper's
contribution sits. Follow citation graphs both directions from the closest hits.

**Verification rules — non-negotiable.**

- Never invent a citation, DOI, arXiv ID, equation number, or quotation. A
  fabricated reference is worse than a missing one, and I will check.
- Tag every entry `[VERIFIED]` (you retrieved the record and confirmed the
  bibliographic data), `[ABSTRACT-ONLY]` (metadata and abstract confirmed, full
  text not read), or `[UNVERIFIED]` (second-hand — say through what).
- Never infer what a paper does from its title or abstract when the claim you
  are making about it is load-bearing. If the full text is paywalled, say so and
  mark it as a gap rather than guessing.
- Quote exactly, with a locator, when the wording carries the argument.
- If a search tool errors or rate-limits, report that rather than silently
  degrading to a weaker source.

## 6. Output

A single markdown document with the six deliverables as top-level sections, in
order. Tables where the brief asks for tables. Put the threat register's most
severe item first and do not bury it.

**Then a final section: "What a hostile reviewer says."** Five to ten of the
sharpest objections you can construct against the paper as described, each with
the strongest available answer — or an honest "no answer, must be conceded".
Include at least one objection to the *evaluation* and one to the *framing*,
not only to novelty.

**Style.** Terse and factual. No padding, no restating the brief back at me, no
enthusiasm. Assume I know the field. If something is uncertain, say how
uncertain and why — a flagged doubt is useful, a confident guess is not. If you
conclude a claimed contribution does not survive contact with the literature,
lead with that; it is the single most valuable thing you could tell me.
