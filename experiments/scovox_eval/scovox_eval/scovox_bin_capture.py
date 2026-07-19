"""ROS 2 node: capture ONE ScovoxMapBinary snapshot and save the full split
Beta/Dir grid to .npz for offline uncertainty analysis (E1-E4).

A fresh subscriber connecting triggers scovox_node to publish a full-map
snapshot (scovox_node.cpp:1608 `snapshot = cur_sub > prev_sub_count_`), and
the QoS here is VOLATILE, so the first message we receive IS that snapshot.

Usage:
  python3 -m scovox_eval.scovox_bin_capture --ros-args \
    -p topic:=/atlas_gt/scovox_node/scovox_bin -p output:=map_bin.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from scovox_msgs.msg import ScovoxMapBinary

# Decoder lives with the offline analysis code.
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "uncertainty"))
sys.path.insert(0, "/home/kalhan/Projects/scovox_ws/experiments/uncertainty")
import scovox_bin  # noqa: E402


class BinCapture(Node):
    def __init__(self):
        super().__init__("scovox_bin_capture")
        topic = self.declare_parameter("topic", "/scovox_node/scovox_bin").value
        self._output = self.declare_parameter("output", "map_bin.npz").value
        # Match the publisher: reliable, KeepLast(50), volatile.
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=50,
                         durability=DurabilityPolicy.VOLATILE)
        self._sub = self.create_subscription(ScovoxMapBinary, topic, self._cb, qos)
        self._done = False
        self.get_logger().info(f"Waiting for ScovoxMapBinary snapshot on {topic} ...")

    def _cb(self, msg: ScovoxMapBinary):
        if self._done:
            return
        try:
            frame = scovox_bin.decode_blob(bytes(msg.data))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"decode failed ({e}); waiting for next msg")
            return
        nb, nd = len(frame["beta"]), len(frame["dir"])
        if nb == 0:
            self.get_logger().warn(f"empty beta frame (dir={nd}); waiting for snapshot")
            return
        self._done = True
        np.savez_compressed(self._output, **scovox_bin.to_npz_dict(frame))
        b = frame["beta"]
        s = b["a_occ"] + b["a_free"]
        p = b["a_occ"] / np.where(s > 0, s, 1.0)
        self.get_logger().info(
            f"Saved beta={nb} dir={nd} -> {self._output} "
            f"(p_occ<.5: {np.mean(p < 0.5):.1%}, >.5: {np.mean(p > 0.5):.1%})")
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = BinCapture()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
