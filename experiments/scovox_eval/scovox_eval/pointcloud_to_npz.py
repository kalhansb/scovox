"""ROS 2 node that subscribes to SCovox PointCloud2 and saves voxels as .npz.

Captures a single snapshot of the map pointcloud, extracts per-voxel fields,
and writes them to a numpy archive for offline evaluation.

Usage:
    ros2 run scovox_eval pointcloud_to_npz --ros-args -p topic:=/robot0/scovox/pointcloud
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import struct


def _read_pc2_fields(msg: PointCloud2) -> dict[str, np.ndarray]:
    """Parse a PointCloud2 message into a dict of named arrays."""
    field_map = {}
    for f in msg.fields:
        field_map[f.name] = (f.offset, f.datatype)

    dtype_map = {
        1: ("b", 1),   # INT8
        2: ("B", 1),   # UINT8
        3: ("h", 2),   # INT16
        4: ("H", 2),   # UINT16
        5: ("i", 4),   # INT32
        6: ("I", 4),   # UINT32
        7: ("f", 4),   # FLOAT32
        8: ("d", 8),   # FLOAT64
    }

    n = msg.width * msg.height
    result = {}
    raw = bytes(msg.data)

    np_dtype_map = {
        "b": np.int8, "B": np.uint8, "h": np.int16, "H": np.uint16,
        "i": np.int32, "I": np.uint32, "f": np.float32, "d": np.float64,
    }
    for name, (offset, datatype) in field_map.items():
        fmt_char, size = dtype_map.get(datatype, ("f", 4))
        arr = np.empty(n, dtype=np_dtype_map.get(fmt_char, np.float32))
        for i in range(n):
            idx = i * msg.point_step + offset
            arr[i] = struct.unpack_from(fmt_char, raw, idx)[0]
        result[name] = arr

    return result


class PointCloudCapture(Node):
    def __init__(self):
        super().__init__("pointcloud_to_npz")
        topic = self.declare_parameter("topic", "/scovox/pointcloud").value
        output = self.declare_parameter("output", "scovox_map.npz").value
        self._output = output
        self._sub = self.create_subscription(PointCloud2, topic, self._cb, 1)
        self._captured = False
        self.get_logger().info(f"Waiting for PointCloud2 on {topic} ...")

    def _cb(self, msg: PointCloud2):
        if self._captured:
            return
        self._captured = True

        fields = _read_pc2_fields(msg)
        points = np.column_stack([fields["x"], fields["y"], fields["z"]])

        save_dict = {"points": points}
        for name in ["occupancy_prob", "semantic_class", "semantic_confidence",
                      "posterior_variance", "eig", "a_occ", "a_free",
                      # E5 — raw mass fields (only present after the 2026-05-06
                      # publish-path patch). Older runs simply lack these keys.
                      "a_unk", "sem_cnt0", "sem_cls0", "sem_cnt1", "sem_cls1"]:
            if name in fields:
                save_dict[name] = fields[name]

        np.savez_compressed(self._output, **save_dict)
        self.get_logger().info(
            f"Saved {len(points)} voxels to {self._output}"
        )
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudCapture()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
