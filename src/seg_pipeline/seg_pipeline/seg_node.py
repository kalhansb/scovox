#!/usr/bin/env python3
"""Online RGB-D segmentation node for the SCovox semantic-mapping pipeline.

Subscribes to a synchronized (depth, color) pair, runs an outdoor semantic
segmentation model on the color frame, and publishes:

  * a colored segmentation `Image` (rgb8)   -> SCovox `seg_topic`
  * the depth `Image` re-stamped + re-framed -> SCovox `depth_topic`
  * the depth CameraInfo re-framed           -> SCovox `depth_info_topic`

ALL THREE carry the DEPTH frame's timestamp and a BODY `frame_id`
(`camera_color_frame`, not the optical frame). That is what makes this robust:

  - depth<->seg always share one stamp, so SCovox's 0.05 s ApproximateTime sync
    matches them no matter how long inference took (lag only lowers map rate);
  - SCovox's hardcoded optical->body `kR` rotation expects a BODY depth frame_id,
    which the re-framed depth provides.

Frames downstream: a LiDAR localizer publishes `map -> ... -> base_link`, and the
bag's `/tf_static` chains `base_link -> camera_link -> camera_color_frame`, so the
re-framed images resolve in `map` for fusion with the LiDAR map.

Model (default): facebook/mask2former-swin-large-mapillary-vistas-semantic
(Mapillary Vistas, 65 outdoor classes) -> collapsed to the compact outdoor set in
outdoor_palette.py. Weights are CC-BY-NC (research use).

Run (inside the GPU container, with a bag playing on /clock):
  python3 -m seg_pipeline.seg_node --ros-args -p use_sim_time:=true
(or, in a sourced colcon workspace: ros2 run seg_pipeline seg_node)
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import message_filters

import torch
import torch.nn.functional as F

from .outdoor_palette import build_collapse_lut, COMPACT_COLORS, COMPACT_NAMES, NUM_CLASSES


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)
INFO_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class SegNode(Node):
    def __init__(self):
        super().__init__("rgbd_seg_node")

        p = self.declare_parameter
        self.depth_topic = p("depth_topic", "/camera/aligned_depth_to_color/image_raw").value
        self.color_topic = p("color_topic", "/camera/color/image_raw/compressed").value
        self.info_topic = p("info_topic", "/camera/aligned_depth_to_color/camera_info").value
        self.out_seg_topic = p("out_seg_topic", "/scovox/segmentation/colored").value
        self.out_depth_topic = p("out_depth_topic", "/scovox/depth/image_raw").value
        self.out_info_topic = p("out_info_topic", "/scovox/depth/camera_info").value
        self.output_frame = p("output_frame", "camera_color_frame").value
        self.model_name = p(
            "model_name",
            "facebook/mask2former-swin-large-mapillary-vistas-semantic",
        ).value
        self.device = p("device", "cuda").value
        self.sync_slop = float(p("sync_slop", 0.05).value)
        self.infer_short_edge = int(p("infer_short_edge", 720).value)  # bound compute
        self.fp16 = bool(p("fp16", True).value)

        if self.device == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA not available — falling back to CPU (very slow).")
            self.device = "cpu"

        self.get_logger().info(f"Loading model: {self.model_name} on {self.device} ...")
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        # Bound the resize so we don't silently upscale to the model's training
        # resolution (Mapillary is large) and blow up VRAM/latency.
        try:
            self.processor.size = {"shortest_edge": self.infer_short_edge,
                                   "longest_edge": max(self.infer_short_edge * 2, 1333)}
        except Exception:
            pass
        self.model = (
            Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name)
            .to(self.device)
            .eval()
        )

        id2label = self.model.config.id2label
        self.collapse_lut = np.asarray(build_collapse_lut(id2label), dtype=np.int32)
        self.color_lut = np.asarray(COMPACT_COLORS, dtype=np.uint8)  # (NUM_CLASSES, 3)
        self._log_class_mapping(id2label)

        # Publishers
        self.seg_pub = self.create_publisher(Image, self.out_seg_topic, SENSOR_QOS)
        self.depth_pub = self.create_publisher(Image, self.out_depth_topic, SENSOR_QOS)
        self.info_pub = self.create_publisher(CameraInfo, self.out_info_topic, INFO_QOS)

        # CameraInfo cached from the latest message
        self.last_info: CameraInfo | None = None
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, INFO_QOS)

        # Synchronized depth + color
        depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=SENSOR_QOS)
        color_sub = message_filters.Subscriber(self, CompressedImage, self.color_topic, qos_profile=SENSOR_QOS)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, color_sub], queue_size=2, slop=self.sync_slop)
        self.sync.registerCallback(self._on_pair)

        self._busy = False
        self._n_in = 0
        self._n_done = 0
        self.create_timer(10.0, self._heartbeat)
        self.get_logger().info(
            f"Ready. in: depth={self.depth_topic} color={self.color_topic} "
            f"info={self.info_topic} | out: seg={self.out_seg_topic} "
            f"depth={self.out_depth_topic} frame={self.output_frame}")

    # ---- callbacks ---------------------------------------------------------
    def _on_info(self, msg: CameraInfo):
        self.last_info = msg

    def _on_pair(self, depth_msg: Image, color_msg: CompressedImage):
        self._n_in += 1
        if self._busy:
            return  # drop-while-busy -> natural throttle, sync stays correct
        self._busy = True
        try:
            rgb = self._decode_color(color_msg)
            if rgb is None:
                return
            H, W = depth_msg.height, depth_msg.width
            seg_model = self._infer(rgb)                      # (h,w) model class ids
            if seg_model.shape != (H, W):
                h0, w0 = seg_model.shape
                ys = np.minimum(np.arange(H) * h0 // H, h0 - 1)
                xs = np.minimum(np.arange(W) * w0 // W, w0 - 1)
                seg_model = seg_model[ys][:, xs]             # nearest-neighbour resize
            compact = self.collapse_lut[seg_model]           # (H,W) compact ids
            color_seg = self.color_lut[compact]              # (H,W,3) rgb8

            stamp = depth_msg.header.stamp
            self.seg_pub.publish(self._make_rgb8(color_seg, stamp))
            self.depth_pub.publish(self._reframe_depth(depth_msg, stamp))
            if self.last_info is not None:
                self.info_pub.publish(self._reframe_info(stamp))
            self._n_done += 1
        except Exception as e:  # keep the node alive on a bad frame
            self.get_logger().error(f"frame failed: {e}")
        finally:
            self._busy = False

    # ---- helpers -----------------------------------------------------------
    def _decode_color(self, msg: CompressedImage):
        try:
            img = PILImage.open(io.BytesIO(bytes(msg.data))).convert("RGB")
        except Exception as e:
            self.get_logger().warn(f"failed to decode compressed color frame: {e}")
            return None
        return np.asarray(img)  # (H, W, 3) uint8 RGB

    @torch.inference_mode()
    def _infer(self, rgb: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device == "cuda" and self.fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                out = self.model(**inputs)
        else:
            out = self.model(**inputs)
        seg = self.processor.post_process_semantic_segmentation(
            out, target_sizes=[(rgb.shape[0], rgb.shape[1])])[0]
        return seg.to("cpu").numpy()

    def _make_rgb8(self, rgb: np.ndarray, stamp) -> Image:
        h, w, _ = rgb.shape
        m = Image()
        m.header.stamp = stamp
        m.header.frame_id = self.output_frame
        m.height, m.width = h, w
        m.encoding = "rgb8"
        m.is_bigendian = 0
        m.step = w * 3
        m.data = np.ascontiguousarray(rgb).tobytes()
        return m

    def _reframe_depth(self, depth_msg: Image, stamp) -> Image:
        m = Image()
        m.header.stamp = stamp
        m.header.frame_id = self.output_frame
        m.height = depth_msg.height
        m.width = depth_msg.width
        m.encoding = depth_msg.encoding
        m.is_bigendian = depth_msg.is_bigendian
        m.step = depth_msg.step
        m.data = depth_msg.data
        return m

    def _reframe_info(self, stamp) -> CameraInfo:
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.output_frame
        src = self.last_info
        info.height = src.height
        info.width = src.width
        info.distortion_model = src.distortion_model
        info.d = src.d
        info.k = src.k
        info.r = src.r
        info.p = src.p
        info.binning_x = src.binning_x
        info.binning_y = src.binning_y
        info.roi = src.roi
        return info

    def _log_class_mapping(self, id2label: dict):
        # Group model classes by destination compact id so the user can sanity-check.
        groups: dict[int, list[str]] = {i: [] for i in range(NUM_CLASSES)}
        for k, name in id2label.items():
            groups[int(self.collapse_lut[int(k)])].append(str(name))
        self.get_logger().info(
            f"Model has {len(id2label)} classes -> {NUM_CLASSES} compact (0=other/unknown).")
        for cid in range(NUM_CLASSES):
            names = groups.get(cid, [])
            self.get_logger().info(
                f"  [{cid:2d}] {COMPACT_NAMES[cid]:11s} <- {len(names)} model cls"
                + (f": {', '.join(names[:6])}{' …' if len(names) > 6 else ''}" if names else ""))
        if not any(groups[c] for c in range(1, NUM_CLASSES)):
            self.get_logger().error(
                "No model classes mapped to ANY real class — id2label may be wrong "
                "(e.g. ImageNet labels). Check the model's config.json.")

    def _heartbeat(self):
        self.get_logger().info(
            f"seg: received {self._n_in} pairs, segmented {self._n_done} "
            f"(dropped {self._n_in - self._n_done} while busy)")


def main():
    rclpy.init()
    node = SegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
