# Registers (remove=false) or unregisters (remove=true) a fine-TSDF
# refinement region on a scovox_node — a vertical cylinder around a trunk
# inside which ray hits additionally fuse into the fine-resolution TSDF
# lattice (docs/design/fine_tsdf_band_dbh_2026_07_30.md). Consumed on the
# node's `fine_region_topic` subscription; ignored when the node runs with
# fine_ratio_log2 = 0.
#
# Field-compatible with explo_planner_msgs/TreeTarget (target_id, center,
# radius), so a one-line relay closes the detect -> orbit -> measure loop.

# Stable region id. Re-sending an id replaces that region's cylinder.
uint32 id

# Trunk axis XY and trunk base z, in the node's integration (map) frame.
# The node derives the refined slab as
#   [z_lo, z_hi] = base_z + [fine_region_z_lo, fine_region_z_hi]   (params).
float32 x
float32 y
float32 base_z

# Estimated trunk radius (m) — the canonical anchor/DBH model. The gate adds
# the node's fine_region_margin. <= 0 is accepted (margin-only gate).
float32 radius

# true = unregister this id instead (already-fused fine voxels persist).
bool remove
