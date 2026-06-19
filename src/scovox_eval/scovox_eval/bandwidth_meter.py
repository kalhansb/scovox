"""Subscribe to ScovoxMapBinary deltas + raw depth images during a fusion
run and accumulate per-source byte counts. Used for E2.1 / F4 bandwidth
characterisation.

Captures three streams:
  - /robotA/scovox_node/scovox_bin  (wire-format deltas — what dscovox sees)
  - /robotB/scovox_node/scovox_bin
  - /robot{A,B}/rgbd_camera_depth_image  (raw sensor proxy for "no Bayesian
    abstraction" baseline)

On Ctrl-C / shutdown emits a JSON summary to --output_json with:
  {robotA_bin_bytes, robotA_bin_msgs,
   robotB_bin_bytes, robotB_bin_msgs,
   robotA_depth_bytes, robotA_depth_msgs,
   robotB_depth_bytes, robotB_depth_msgs,
   start_wallclock, end_wallclock}
"""
import json
import signal
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from scovox_msgs.msg import ScovoxMapBinary


class BandwidthMeter(Node):
    def __init__(self, output_json: Path):
        super().__init__("bandwidth_meter")
        self.output_json = output_json
        self.start_t = time.time()

        self.bin_bytes = {"robotA": 0, "robotB": 0}
        self.bin_msgs = {"robotA": 0, "robotB": 0}
        self.depth_bytes = {"robotA": 0, "robotB": 0}
        self.depth_msgs = {"robotA": 0, "robotB": 0}

        bin_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        depth_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
        )

        for r in ("robotA", "robotB"):
            self.create_subscription(
                ScovoxMapBinary,
                f"/{r}/scovox_node/scovox_bin",
                lambda msg, _r=r: self._on_bin(msg, _r),
                bin_qos,
            )
            self.create_subscription(
                Image,
                f"/{r}/rgbd_camera_depth_image",
                lambda msg, _r=r: self._on_depth(msg, _r),
                depth_qos,
            )

        self.get_logger().info(
            f"bandwidth_meter listening: scovox_bin × 2 + depth × 2 → {output_json}"
        )

        # Dump every 2 s so even a SIGKILL leaves the latest snapshot on disk.
        # rclpy.spin() blocks Python signal delivery, so the SIGINT/SIGTERM
        # handler often never fires — the timer is the reliable path.
        self.create_timer(2.0, self.dump)

    def _on_bin(self, msg: ScovoxMapBinary, robot: str):
        # Payload bytes only — count what's actually shipped on the wire.
        self.bin_bytes[robot] += len(bytes(msg.data))
        self.bin_msgs[robot] += 1

    def _on_depth(self, msg: Image, robot: str):
        self.depth_bytes[robot] += len(bytes(msg.data))
        self.depth_msgs[robot] += 1

    def dump(self):
        end_t = time.time()
        out = {
            "start_wallclock": self.start_t,
            "end_wallclock": end_t,
            "duration_s": end_t - self.start_t,
        }
        for r in ("robotA", "robotB"):
            out[f"{r}_bin_bytes"] = self.bin_bytes[r]
            out[f"{r}_bin_msgs"] = self.bin_msgs[r]
            out[f"{r}_depth_bytes"] = self.depth_bytes[r]
            out[f"{r}_depth_msgs"] = self.depth_msgs[r]
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output_json.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        tmp.replace(self.output_json)


def main():
    rclpy.init()
    output_json = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/bandwidth.json")
    node = BandwidthMeter(output_json)

    def _sigint(*_a):
        node.dump()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)
    try:
        rclpy.spin(node)
    finally:
        node.dump()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
