# Per-source integration counters for the fused map: what each peer has
# delivered into this robot's fusion, and how much of the fused map that
# writing touched. Published by dscovox on its own timer.
#
# WHY THIS EXISTS. Nothing downstream could previously answer "have the maps
# exchanged?". The planner's own map total (total_observed_voxels) sums this
# robot's own sensing and every peer's contribution into one number, so a
# robot standing at a meeting cannot tell a partner that has finished sending
# from a partner whose bytes are not arriving at all. Both look like a total
# that has stopped moving. These counters split that number by source so the
# question has an answer.
#
# ALL ARRAYS ARE PARALLEL and indexed together; source_frame[i] names the robot
# that counters[i] describe. There is one entry per source dscovox has ever
# heard from, in sorted source_frame order so successive samples line up
# without a join. A source that has gone quiet keeps its entry with its counts
# frozen — it is never removed, because removing it would make "silent" and
# "never seen" the same observation.
#
# PUBLISHED UNCONDITIONALLY, on a fixed timer, whether or not anything arrived
# and whether or not the fused map changed. This is the whole point and it is
# the opposite of how the fused map itself publishes (that is dirty-gated). A
# consumer must be able to distinguish "this peer sent nothing since the last
# sample" from "dscovox has not told me anything lately", and it can only do
# that if a silent interval still produces samples. A counter whose silence is
# ambiguous is not an observation.
#
# MONOTONE WITHIN A NODE LIFETIME, AND NOT ACROSS ONE. Both counters only ever
# increase while dscovox runs. They are node-local and reset to zero if it
# restarts, so a consumer differencing them against a captured baseline must
# treat a DECREASE as a re-baseline, never as arrival.
#
# header.stamp = the publish instant on dscovox's clock. Diagnostic: consumers
#   difference the counters against their own captured baseline and time the
#   interval on their own clock, exactly as they do for the cell census.
# header.frame_id = the fused map frame (e.g. "map").
std_msgs/Header header

# The source robot's integration frame, as carried in ScovoxMapBinary's
# header.frame_id (e.g. "atlas/odom"). This is the key dscovox files per-source
# grids under, and it survives relaying: the comms emulator forwards serialized
# bytes without deserializing, so a peer's update credits the robot that sensed
# it rather than whichever robot bridged it.
string[] source_frame

# Cumulative count of voxel deltas RECEIVED from this source and ingested into
# its grid: occupancy plus semantics, summed. "Did anything cross the radio
# from this robot?"
#
# THIS IS THE ARRIVAL TERM AND IT IS THE LOAD-BEARING ONE. A consumer deciding
# whether a peer has finished sending must check that this rose above the value
# it had when the meeting started, not merely that it has stopped rising: a
# peer that never sent a byte and a peer that has sent everything it has both
# present a rate of zero, and only the difference against a baseline separates
# them. Counting arrival rather than effect is deliberate for the same reason —
# a delivery that changed nothing still proves the channel is carrying.
#
# Frames rejected before ingest (bad endianness, failed decompression, a
# mismatched Dirichlet prior) are NOT counted. They prove the link carried
# bytes but they will never contribute map, and a peer whose every frame is
# rejected is a configuration fault, not a partner worth waiting for.
uint64[] deltas_received

# Cumulative count of fused-map cells WRITTEN while integrating this source:
# occupancy plus semantics, summed, de-duplicated within each frame.
#
# CELLS WRITTEN, NOT CELLS CHANGED. The ingest path snapshot-replaces and
# records the coordinate unconditionally; it does not compare against the
# previous value. So this does not measure novelty, and it does not fall to
# zero while a peer keeps re-sending ground both robots already hold. What it
# measures is how much of the fused map this peer's traffic reached, which is
# the "did the map actually move" half of the exchange question and the reason
# both numbers are published instead of just the arrival count.
uint64[] cells_touched
