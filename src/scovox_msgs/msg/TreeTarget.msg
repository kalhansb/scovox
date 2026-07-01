# A single tree target to be perceptively exploited (circled + dwelled at
# occlusion-free vantage points). This is the shared seam between target
# *production* and the planner's exploitation behaviour: today a time-based
# scheduler publishes a preselected list on this message, but a real tree
# detector can publish the identical message later with zero planner changes.
# Consumed by explo_planner (see explo_planner/include/explo_planner/target_queue.hpp).
#
# header.stamp    = release / detection time (sim time during bag playback)
# header.frame_id = the map frame the centre lives in (e.g. "map")
std_msgs/Header header

# Stable target id. Used by the planner to dedup re-reports and to track
# per-target completion status. Two messages with the same id are the same
# tree; the planner keeps the first and ignores later duplicates.
uint32 target_id

# Trunk centre in the map frame. XY is the trunk axis; z is the trunk base
# (ground contact). Vantage points are generated on a circle in the XY plane
# around (x, y).
geometry_msgs/Point center

# Estimated trunk radius (m). Sets the vantage standoff distance
# (standoff = radius + vantage_standoff_m). <= 0 => use the default standoff
# only (treat the trunk as a point).
float32 radius

# Optional trunk vertical extent (m), informational. <= 0 => the planner uses
# its configured ROI z-band. Reserved for height-aware vantage tilt later.
float32 height

# Provenance of this target. "schedule" when emitted by the time-based
# scheduler; a robot id (e.g. "atlas") when a detector running on that robot
# feeds it. Diagnostic only.
string discovered_by

# Target lifecycle status. 0 = PENDING (awaiting / undergoing exploitation),
# 1 = DONE. Reserved for a future multi-robot completion broadcast so peers
# can drop a tree another robot already finished; the single-robot planner
# tracks completion in its local queue and does not require this field.
uint8 STATUS_PENDING = 0
uint8 STATUS_DONE = 1
uint8 status
