std_msgs/Header header

# Binary encoding of the SCovox map (incremental delta update format).
# header.frame_id identifies the source robot (its integration frame, e.g.
# "atlas/odom"). The merger keys per-source grids by this frame.
uint8 version
bool little_endian

# Transform taking a point expressed in header.frame_id (the source robot's
# integration frame) into the shared "map" frame. Captured by the producer at
# publish time and carried with the update so the merger never needs the
# source->map transform from its own TF tree. CDR handles endianness for this
# field (unlike the raw `data` blob below).
geometry_msgs/Transform map_from_source

# Raw serialized bytes (LZ4 compressed)
uint8[] data
