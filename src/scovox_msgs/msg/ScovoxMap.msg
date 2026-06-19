# Complete Voxel Map Message
# Represents the entire SCovox map state

# Standard ROS message header
std_msgs/Header header

# Map metadata
float32 resolution              # Voxel resolution in meters
geometry_msgs/Point32 origin    # Map origin in world coordinates (float32)

# Map parameters
float32 occupancy_threshold     # Threshold for occupancy classification
float32 semantic_threshold      # Threshold for semantic confidence
uint8 max_semantic_classes     # Maximum number of semantic classes

# Voxel data
ScovoxVoxel[] voxels            # Array of all voxels in the map