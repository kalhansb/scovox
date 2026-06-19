std_msgs/Header header

# Binary encoding of the SCovox map
# version 1: scovox_mapping_binary incremental update format (float evidence)
# header.frame_id identifies the source robot (its integration frame, e.g.
# "atlas/odom"). The merger keys per-source grids by this frame.
uint8 version
bool little_endian

# Raw serialized bytes (LZ4 compressed)
uint8[] data
