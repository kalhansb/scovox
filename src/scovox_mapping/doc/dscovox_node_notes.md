# dscovox_node.cpp — design notes and history

The long comments of `src/dscovox_node.cpp`, moved out of the code on 2026-09-23 so the source carries short comments only. Where a comment was moved, the code keeps a short gist ending in `(notes: <id>)`; the section headed `<id>` below holds the original comment, word for word.

Sections follow the order of the source file and are grouped by the function (or section) they sit in. Each gives the line of code the comment was attached to and its original line number. Line numbers, dates, generation numbers and cross-references inside the moved text are as they were when written; they record history and are not maintained.

## Contents

- [File scope](#file-scope) — 4
- [SourceGrid — declarations](#sourcegrid--declarations) — 3
- [File scope (part 2)](#file-scope-part-2) — 1
- [DSCovoxNode](#dscovoxnode) — 8
- [rclcpp::create_timer](#rclcppcreate_timer) — 1
- [DSCovoxNode — declarations](#dscovoxnode--declarations) — 1
- [onBinaryMap](#onbinarymap) — 4
- [DSCovoxNode — declarations (part 2)](#dscovoxnode--declarations-part-2) — 1
- [maybePublishPointCloud](#maybepublishpointcloud) — 1
- [forEachCell](#foreachcell) — 2
- [DSCovoxNode — declarations (part 3)](#dscovoxnode--declarations-part-3) — 2
- [publishFusedMap](#publishfusedmap) — 1
- [DSCovoxNode — declarations (part 4)](#dscovoxnode--declarations-part-4) — 4

## File scope

### dscovox-local-to-fused-topology

**One-way local to fused topology** — attached to `#include <rclcpp/rclcpp.hpp>` (line 3)

```text
Each connected scovox_node ships ScovoxMapBinary deltas of its persistent
LOCAL scovox grid (i.e. that robot's own sensor observations only — NEVER
the fused dscovox grid). This one-way local→fused topology is what keeps
the additive Beta–Dirichlet consensus Bayesian: there is no comms-level
echo where one robot's evidence is shipped back into another robot's
source grid via the merged map, so conditional independence of the two
sources given the latent voxel state holds at the protocol level. The
dscovox `~/scovox` publish is for downstream consumers (planner,
visualisation) only — wiring it back as an input to another dscovox would
re-introduce evidence-echo and break the additive rule.
```

### dscovox-source-grids-map-frame

**Source grids stored in map frame** — attached to `#include <rclcpp/rclcpp.hpp>` (line 19)

```text
The merger keeps one source grid per robot keyed by the binary's
header.frame_id (e.g. "atlas/odom"). Source grids are stored directly in
MAP-FRAME coordinates: at receive time we transform each delta voxel once
using the source->map pose the producer captured and CARRIED in the message
(ScovoxMapBinary.map_from_source), cached the first time we see that source.
This node holds NO TF listener — the pose rides with the data.
```

### dscovox-requires-cslam-disabled

**Why c-slam must stay disabled** — attached to `#include <rclcpp/rclcpp.hpp>` (line 27)

```text
The cached source->map pose is never refreshed (first carried pose wins).
This is correct only while TFs are static. Re-enabling c-slam (loop closures
/ pose-graph optimization) is a CORRECTNESS BUG: the first pose jump leaves
every voxel in the source grid at its old map-frame coord, producing ghost
voxels at pre-loop-closure positions and missing voxels at the new positions.
Before turning c-slam back on, refactor SourceGrid to store evidence in
source-frame coords + reproject on each carried-pose change, with a
per-source "pose changed → reproject" handler. See ablation entry C5 in
docs/issues/ablations_punch_list.md for the design.
```

### dscovox-reset-then-refold

**Incremental reset-then-refold of fused cells** — attached to `#include <rclcpp/rclcpp.hpp>` (line 37)

```text
On every binary we incrementally update the fused grid by, for each touched
map-frame coord, resetting fused[c] to the prior and re-folding the current
state of every source's grid at c. The fold uses the same Beta-conjugate
consensus as before: a_fused = a_1 + a_2 - 1. Cells the binary did not
touch are not visited — their existing fused value is still correct because
no source's contribution at those cells changed.

The reset-then-refold pattern is what keeps this bit-for-bit equivalent to
a from-scratch rebuild while making the work proportional to the delta size
instead of the total map size. Critically, it cannot double-count: a
source's previous contribution at a cell is wiped before that source's
current contribution is folded back in.
```

## SourceGrid — declarations

### source-grid-split-grids

**Per-source split Beta/Dir grids** — attached to `std::unique_ptr<Bonxai::VoxelGrid<scovox::BetaVoxel>> beta_grid;` (line 89)

```text
This source's contribution to the world, stored in MAP-FRAME coords.
Each entry is the latest snapshot of (occupancy, semantics) received for
that map-frame voxel from this robot.

Split Beta/Dirichlet (wire format) receiver populates these two grids:
occupancy (BetaVoxel) ∥ semantics (DirVoxel). No TsdfMap on the receiver
because share_tsdf=false is the wire default — TSDF state never crosses the
wire to dscovox in the production path.
```

### source-grid-cached-pose

**Cached static source-to-map transform** — attached to `Eigen::Isometry3d T_map_source{Eigen::Isometry3d::Identity()};` (line 99)

```text
Cached static source->map transform. Taken from the first update's carried
map_from_source pose and never refreshed — this assumes TFs are static
(c-slam disabled). Under c-slam, loop closures change this transform and the
cache becomes a correctness bug. See the file-header banner and C5 in
ablations_punch_list.md.
```

### source-grid-fusion-counters

**Per-source delivery counters** — attached to `uint64_t deltas_received{0};   // voxel deltas ingested from this source` (line 106)

```text
What this source has delivered, cumulatively, since this node started.
Published as ScovoxFusionCounters so a consumer can ask "have the maps
exchanged with THIS robot?" — a question the fused map's own voxel total
cannot answer, because it sums own sensing and every peer into one number.
See the message for what each counts and why arrival is the load-bearing
one. Written under the unique_lock in the ingest path; read under a shared
lock from the publish timer.
```

## File scope (part 2)

### dscovox-consensus-helpers-shared

**Consensus helpers shared with tests** — attached to `} // namespace` (line 129)

```text
The pure receiver-side consensus helpers — kPriorSlop, isPriorBeta/isPriorDir,
projectBetaDirToSemBetaForViz, projectBetaDirToVoxel, and the refoldBeta/
refoldDir cores — now live in scovox/dscovox_consensus.hpp (namespace scovox)
so the unit tests can exercise the SAME code this node runs (findings
#18/#19/#20). The unqualified call sites below resolve to them via ADL (every
call passes a scovox-typed voxel argument).
```

## DSCovoxNode

### dscovox-share-roi-z-band

**Receive-side shared-ROI z-band** — attached to `share_z_min_ = declare_parameter<double>("share_roi_z_min", 0.0);` (line 187)

```text
Shared-ROI z-band — receive-side defensive mirror of the sender's wire
filter. Records whose MAP-frame voxel centre falls outside
[share_roi_z_min, share_roi_z_max] are dropped at ingest, so the fused
map honours the band even if one sender was launched without it.
KEEP IN SYNC with scovox_node share_roi_z_min/share_roi_z_max (sender
wire filter, applied in its integration frame) and with explo_planner
shared_params.yaml roi_min_z/roi_max_z: the shared band must be a
SUPERSET of the planner band. min >= max (default 0/0) disables.
```

### dscovox-global-planning-map

**World-fixed global planning map** — attached to `pub_plan_glob_ = declare_parameter<bool>("publish_global_planning_map", false);` (line 199)

```text
The merger already answers GetOccupancyGrid, but that projection is
sized to a TIGHT BOUNDING BOX of the observed voxels: its origin and
extent move every time the map grows. An exploration planner cannot use
that envelope. It indexes the grid with raw world XY and treats every
out-of-bounds cell as OCCUPIED (isCellOccupied), so a shrink-wrapped
grid reports the entire unexplored world as blocked — candidates are
rejected as unreachable and the planner starves. This publisher is the
same 2D projection over a FIXED envelope that does not move under the
consumer, mirroring scovox_node's own local/global planning-map split.

Two consumers now: the exploration planner, and — since generation 5 —
simple_nav_3d's global nav planner, which was repointed here from
/<robot>/dscovox_node/planning_map. That name never had a publisher, so
the nav global planner sat inert for the whole campaign history before
the repoint. Deliberately still NOT ~/planning_map: that name means the
rolling local crop from scovox_node, and the local nav planner needs it
to keep meaning exactly that.

Off by default: nothing that does not ask for it pays the projection.
```

### dscovox-pointcloud-qos

**Fused pointcloud QoS** — attached to `pc_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(` (line 237)

```text
Explicit reliable + depth 1, mirroring scovox_node's pc_pub_: the fused
cloud is tens of MB (one point per fused voxel) and SystemDefaultsQoS
resolved to BEST_EFFORT on this build — every fragmented sample dropped
and RViz never received a single cloud. Depth 1 caps publisher-side
buffering (see scovox_node's OOM note).
```

### dscovox-fused-map-topic

**Fused-map topic and its latched QoS** — attached to `scovox_map_pub_ = create_publisher<scovox_msgs::msg::ScovoxMap>(` (line 246)

```text
Fused-map topic for downstream consumers (planner, visualisation). Topic
form of the on-demand GetRegion service: a full snapshot of the fused
Beta/Dir grids projected to a ScovoxMap, published from the publish timer
below when the map has changed and someone is subscribed (see
publishFusedMap). Latched QoS — KeepLast(1) + reliable + transient_local —
so the last published snapshot is retained and replayed to a (re)connecting
subscriber; the very first subscriber receives it on the next publish tick
after connecting. Only one full snapshot is ever retained (the voxel dump
can be large). Any subscriber MUST match this QoS or it receives nothing.
```

### dscovox-global-planning-map-qos

**Global planning map QoS** — attached to `{` (line 259)

```text
Latched, matching the exploration planner's subscriber QoS exactly
(KeepLast(1) + reliable + transient_local). The planner blocks in its
start-up wait until the first map arrives, so a QoS mismatch here would
present as a planner that never leaves INIT rather than as an error.
The topic name is declared unconditionally so `ros2 param get` reports
it even when the publisher is disabled.
```

### dscovox-fusion-counters-timer

**Fusion counters on their own timer** — attached to `fusion_counters_pub_ =` (line 279)

```text
Per-source integration counters — "have the maps exchanged, and with
whom?". Reliable and latched for the same reason as the fused map: a
consumer that connects late must receive the current totals rather than
wait for the next tick to learn a peer exists.

ITS OWN TIMER, NOT THE PUBLISH TIMER, and that is the point rather than
an implementation detail. Everything on publish_timer_ is gated on the
fused map having changed (publishFusedMap consumes fused_dirty_) or on
having subscribers, so all of it falls silent in exactly the case a
consumer most needs a sample: nothing is arriving from a peer. Hanging
these counters off that timer would make "this peer is quiet" and
"dscovox is quiet" the same observation, which is the ambiguity the
counters exist to remove. Rate is independent of publish_rate_hz for the
same reason.
```

### dscovox-bin-sub-qos-depth

**Binary delta subscription QoS depth** — attached to `const int bin_depth = std::max(` (line 308)

```text
Reliable + deeper queue for binary submap deltas. The old
SystemDefaultsQoS resolved to BEST_EFFORT on this build, which
silently dropped large ScovoxMapBinary payloads under any
backpressure. Combined with scovox_node's fire-and-forget
dirty_.clear() after publish, drops became permanent voxel loss.
KeepLast absorbs publish bursts when this node is busy
rebuilding the fused grid; reliable forces redelivery of any
packet the transport drops.

Depth is a parameter because the binding limit is the SHALLOWEST end of
the chain, and under a comms emulator the sender end is not the direct
publisher: an outage queues deltas, and reconnect releases the whole
backlog in one pass, far faster than this node drains it. A reliable
KEEP_LAST reader that overflows discards the excess with no error and no
counter — the relay's drop_overflow only sees its own pre-relay queue —
so the loss surfaces as permanently missing voxels in the fused map with
nothing anywhere recording that it happened. Size it to the emulator's
rx_qos_depth (or larger) for those runs. Default 50 = prior behaviour.
```

### dscovox-publish-timer-lock

**One shared lock per publish tick** — attached to `publish_timer_ = rclcpp::create_timer(` (line 344)

```text
The fused grid is kept current incrementally inside onBinaryMap.
The visualization pointcloud is normally driven from there too;
this timer is the fallback (when no binaries arrive).

One outer shared_lock spans every publisher in the tick so they all
see the same fused state — without it, an onBinaryMap callback can
mutate the fused grids between any two of them. publish* helpers
must NOT take mu_ themselves — std::shared_mutex is non-recursive
so re-locking here would be UB.
```

## rclcpp::create_timer

### dscovox-fused-map-gated-on-pub-hz

**Fused-map topic gated on the publish timer** — attached to `publishFusedMap();` (line 360)

```text
Topic form of GetRegion: publish the whole fused map for the
planner. Shares this tick's shared_lock (publishFusedMap must not
re-lock the non-recursive shared_mutex). NOTE: gated on pub_hz_ > 0
like every other publisher here — disabling the publish timer also
disables the fused-map topic the planner depends on.
```

## DSCovoxNode — declarations

### dscovox-receive-path

**The split Beta/Dir receive path** — attached to `void onBinaryMap(const scovox_msgs::msg::ScovoxMapBinary::SharedPtr msg) {` (line 404)

```text
ScovoxMapBinary receive path — the node's only receive path. Every binary
is a split Beta/Dir envelope. Operates on the de-unified BetaVoxel
(occupancy) ∥ DirVoxel (semantics) grids + consensus_merge.hpp. The two
grids ingest + refold INDEPENDENTLY: a touched coord may be in either or
both. Priors are pinned from the first frame's header.

NOTE: the RPC query services (GetRegion / GetOccupancyGrid) project the
split Beta(+Dir) grids into a transient scovox::Voxel via a
substrate-agnostic templated core. The SEMANTIC query math uses the
raw-evidence convention; the OCCUPANCY math uses the symmetric Beta(1,1)
prior (p_occ=0.5) — see projectBetaDirToVoxel / docs/occupancy_prior.md.
Occupancy-only services (GetOccupancyGrid) read just the Beta grid;
GetRegion joins the Dir grid for per-class evidence.
```

## onBinaryMap

### dscovox-endianness-reject

**Cross-endian frames are rejected** — attached to `constexpr bool kHostLittleEndian =` (line 425)

```text
The frame body is serialized in host byte order (BinarySerializer uses raw
field memcpy, not an endian-canonical encoding). The publisher stamps
msg->little_endian from its host; until a byte-swapping decode path exists,
a sender of the opposite endianness can only be mis-decoded. Reject it loud
rather than silently corrupting the fused map. (No-op on a homogeneous
little-endian fleet, which is every supported target today.)
```

### dscovox-carried-pose-first-wins

**Carried pose, first pose wins** — attached to `const Eigen::Isometry3d Tmo = tfToIsometry(msg->map_from_source);` (line 461)

```text
The producer captured its source->map pose at publish time and carried it
in the message, so the merger never touches TF. First-pose-wins: the first
frame from a source pins T_map_source (below); later frames' carried poses
are ignored. The static-TF / c-slam-off assumption is unchanged — only its
mechanism moved from a TF lookup to the message's map_from_source.
```

### dscovox-dir-fold-order

**Deterministic Dir fold order** — attached to `std::vector<const std::string*> keys;` (line 605)

```text
Fold the Dir sources in a deterministic (sorted-by-source-id) order.
mergeDir truncates to top-K and a class dumped to OTHER cannot climb
back, so the fused slots — and hence dominantClass / mesh labels —
depend on fold order. Iterating sources_ (an unordered_map) directly
would let them flip across runs and rehashes; sort the keys first so
the refold is reproducible. (Beta merge is additive/commutative and
needs no ordering.)
```

### dscovox-counters-presented-not-stored

**Counters count deltas as presented** — attached to `src.deltas_received += n_beta_deltas + n_dir_deltas;` (line 630)

```text
Accumulate this source's running totals while the write lock is still
held. Deliberately AFTER the reject paths above: a frame that returned
early (bad prior, num_classes==0) never became map and is not counted
as delivery.

ARRIVAL IS COUNTED AS PRESENTED, NOT AS STORED — the deltas the frame
carried, before the shared-ROI z-band clip drops the ones outside the
band. That is the right measure for the question this counter answers:
a clipped delta still crossed the radio, and the clip is this robot's
own policy rather than anything about whether the peer is reaching it.
cells_touched, being about the fused map, is counted post-clip.
```

## DSCovoxNode — declarations (part 2)

### dscovox-maybe-publish-pointcloud

**Rate-limited pointcloud publish** — attached to `void maybePublishPointCloud() {` (line 707)

```text
Rate-limited visualization publish. Called from the binary callback's
tail (so the user-visible map updates as soon as ingest produces fresh
data) and from the timer as a fallback (so the map keeps refreshing in
RViz even if no binaries arrive). publishPointCloud already returns
early if no subscriber, so this is free when nobody is watching.

Caller must hold mu_ (shared). The lock is hoisted to the call sites so
every publisher in a single timer tick sees the same fused state.
```

## maybePublishPointCloud

### dscovox-pointcloud-cas-rate-limit

**Race-free pointcloud rate limit** — attached to `const int64_t now_ns = get_clock()->now().nanoseconds();` (line 716)

```text
Race-free rate limit: this runs under only a SHARED lock and can execute
concurrently from the timer and a binary callback. Claim the window with a
single atomic compare_exchange so exactly one caller proceeds per interval
(a plain read-then-write of a non-atomic timestamp would be a data race
and could let both publish in the same window). See last_pc_pub_ns_.
```

## forEachCell

### dscovox-pointcloud-prior-gate

**Prior-only cells skipped by isPriorBeta** — attached to `if (isPriorBeta(v, fused_num_classes_, fused_alpha_0_)) return;` (line 746)

```text
Skip prior-only cells via isPriorBeta, independent of the prior's p_occ.
(Gating on isPriorBeta rather than a p_occ threshold keeps this correct
for any prior: the old calibrated prior p_occ ≈ 0.933 exceeded the 0.7
threshold and would publish as phantom occupied; the prior is now
Beta(1,1)/0.5. See docs/occupancy_prior.md.) Mirror the RPC walkers' gate.
```

### dscovox-semantic-class-uint8

**semantic_class is a UINT8 field** — attached to `const uint8_t bc = (best_cls < 256) ? static_cast<uint8_t>(best_cls) : 0;` (line 794)

```text
The semantic_class PointField is UINT8 (wire/schema locked — RViz and
pointcloud_to_npz.py read it as one byte), but argmaxClassConfidence
returns a uint16_t class id. A naive static_cast<uint8_t> of an id >=256
(e.g. a 360-class taxonomy) would silently alias to id%256 and collide
with an unrelated class for BOTH the label and the palette colour. Emit
0 (unknown) instead so the >255 case is unambiguous rather than wrong.
Mirror this same UINT8 limit in scovox_node.cpp's pointcloud publishers.
```

## DSCovoxNode — declarations (part 3)

### dscovox-fill-region-lock-free

**Lock-free GetRegion core** — attached to `void fillRegion(const scovox_msgs::srv::GetRegion::Request::SharedPtr& rq,` (line 856)

```text
Lock-free GetRegion core, shared by the service handler and the fused-map
topic publisher. Caller MUST already hold (at least) a shared lock on mu_:
std::shared_mutex is non-recursive, so this function must never lock it
itself. Fills rs->map header + the voxels inside rq's bbox (regionOnGrid
does the clip) by projecting the fused split Beta/Dir grids.
```

### dscovox-publish-fused-map

**Publishing the whole fused map** — attached to `void publishFusedMap()` (line 891)

```text
Publish the ENTIRE fused map as a ScovoxMap topic. Called from the publish
timer, which already holds a shared lock on mu_ — this must NOT lock mu_
(non-recursive shared_mutex). Skipped when no one is subscribed (so a large
voxel dump isn't built for nothing) and when the map hasn't changed since
the last publish. The subscription check comes first so dirtiness is
preserved while nobody listens, then delivered on the first tick after a
subscriber connects. The latched (transient_local) QoS retains that last
published snapshot and replays it to any later (re)connecting subscriber.
```

## publishFusedMap

### dscovox-full-coverage-bbox

**Full-coverage bbox sized by resolution** — attached to `const double big = 1e8 * std::max(static_cast<double>(res_), 1e-3);` (line 904)

```text
Full-coverage bbox: regionOnGrid clips in coord space via posToCoord
(floor(corner/res_)). Size the corner relative to res_ so it lands at
±1e8 voxels — far outside any real map yet comfortably inside int32 at
ANY resolution (a fixed metric constant would overflow the cast at very
fine resolutions). Consumers re-apply their own ROI clip on ingest.
```

## DSCovoxNode — declarations (part 4)

### dscovox-fusion-counters-no-gates

**Fusion counters publish with no gates** — attached to `void publishFusionCounters()` (line 918)

```text
Publish the per-source integration counters. Driven by its own timer, and
unlike every other publisher here it takes its own shared lock, has no
dirty gate, and does not check the subscription count.

NO GATES, ON PURPOSE. The message is a handful of integers, so the cost of
publishing one nobody reads is nothing, while the cost of NOT publishing is
that a consumer cannot tell a peer that has gone quiet from a dscovox that
has gone quiet — and those two need different responses. Every skip
condition the other publishers carry (map unchanged, nobody subscribed)
correlates with exactly the silence this is here to make legible.

Sorted by source frame so successive samples are positionally comparable
(sources_ is an unordered_map and its iteration order is not stable across
rehashes). Entries are never removed: a source that stops sending keeps its
frozen counts, because dropping it would restore the ambiguity between
"silent" and "never heard of".
```

### dscovox-global-map-three-state

**Three-state global planning map contract** — attached to `void publishGlobalPlanningMap() {` (line 956)

```text
World-fixed 2D projection of the FUSED grid for the exploration planner.
Caller must already hold mu_ (shared) — the publish timer does.

Why this does not reuse occupancyGridOnGrid below: that one derives its
origin and extent from the data (a tight bbox), which is correct for an
on-demand service and fatal for a planner that reads out-of-bounds as
occupied. Here the envelope is a constant of the run.

THE THREE-STATE CONTRACT, which is the whole point of the fix:
  -1  unknown  — no evidence. Cells outside the observed set keep this,
                 and so do prior-only voxels (isPriorBeta): a voxel that
                 exists in the grid carrying nothing but its Dirichlet
                 prior has never been measured by anyone. Marking those 0
                 would fabricate free space; marking them 100 would
                 reproduce the starvation this fix exists to remove.
   0  free      — observed, p_occ below the occupancy threshold.
 100  occupied  — observed, p_occ at or above it. Occupied wins ties
                  across voxels sharing a cell (the else-if below).
```

### dscovox-last-pc-pub-atomic

**Atomic pointcloud rate-limit timestamp** — attached to `std::atomic<int64_t> last_pc_pub_ns_{0};` (line 1122)

```text
Rate-limiter timestamp for the visualisation pointcloud, stored as raw
nanoseconds in a std::atomic. maybePublishPointCloud() runs under only a
SHARED lock (from both the binary-callback tail and the publish timer), so
two readers can execute concurrently; a plain rclcpp::Time would be torn /
race-written here. The atomic + compare_exchange below makes the
read-decide-update a single race-free claim so exactly one caller publishes
per window even under a multi-threaded executor.
```

### dscovox-prior-pinned-flag

**Dedicated prior-pinned flag** — attached to `bool     prior_pinned_{false};` (line 1146)

```text
Dedicated "prior has been pinned" flag for the wire receive path. We must
NOT overload fused_num_classes_==0 as the "not yet pinned" sentinel: a
sender that ships num_classes==0 (misconfig/upstream bug) would store 0 and
re-pin/re-log every frame, and the cross-prior mismatch guard would never
run (so a second num_classes==0 source with a different alpha_0 would fuse
without rejection). num_classes==0 frames are rejected explicitly in the path.
```
