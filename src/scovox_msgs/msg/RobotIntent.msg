# Multi-robot exploration claim. Each planner publishes its currently
# selected viewpoint so peers can avoid contesting the same hotspot.
# Used by Coordination::onIntent / claimMatching / selfWinsAgainst
# (see explo_planner/coordination.hpp).
#
# header.stamp = sim time of the claim
# header.frame_id = the planning map frame (e.g. "map")
std_msgs/Header header

# Producer robot id (e.g. "atlas"). Used by peers to drop self-messages
# and as the lex tiebreak key in MinPos when straight-line distances tie.
string robot_id

# Claimed viewpoint position (XY in the map frame; z = candidate robot z).
geometry_msgs/Point goal_pos

# Planned final yaw at goal_pos (radians).
float32 yaw

# Producer's current pose at the time of the claim. MinPos compares
# (peer.robot_pos -> candidate) against (self.pos -> candidate); we ship
# the producer's pose rather than a precomputed cost so peers can recompute
# the comparison locally with their own metric.
geometry_msgs/Point robot_pos

# Disc radius (m) around goal_pos that the producer reserves. Peers treat
# any candidate within this radius as "contested by the producer".
float32 claim_radius_m

# Time-to-live for this claim (sim seconds). Peers drop the claim from
# their local table once header.stamp + ttl_sec is in the past, so a
# stalled or crashed peer eventually releases its target.
float32 ttl_sec

# Diagnostic: which planner produced this claim.
# 0 = eig, 1 = entropy, 2 = frontier, 3 = random
uint8 planner_type
