# ScovoxVoxel Message
# Represents a single voxel with occupancy and semantic information

# Voxel position in 3D space (float32 per component to reduce size)
geometry_msgs/Point32 position

# Occupancy information (derive probability from a_occ/a_free on the consumer side)
float32 a_occ                   # Beta distribution parameter for occupied evidence
float32 a_free                  # Beta distribution parameter for free evidence

# Semantic information
# (derive best class/confidence from semantic_evidence; unknown in a_unk)
float32 a_unk                   # Unknown semantic evidence count

# Raw semantic evidence (top-K classes)
ScovoxSemanticEvidence[] semantic_evidence