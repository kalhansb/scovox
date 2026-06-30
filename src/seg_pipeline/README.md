# seg_pipeline — online RGB-D segmentation for SCovox

Live semantic segmentation of the RealSense color stream, feeding SCovox's RGB-D
semantic-mapping path. Runs in its own GPU container and talks to the scovox +
localizer containers over the shared DDS graph (host net + `ROS_DOMAIN_ID`).

## What it does

`seg_node.py` synchronizes a **(depth, color)** pair, runs an outdoor semantic
model on the color frame, and publishes — all stamped with the **depth frame's
timestamp** and a **body `frame_id`** (`camera_color_frame`):

| out topic (default) | type | for SCovox param |
|---|---|---|
| `/scovox/segmentation/colored` | `sensor_msgs/Image` rgb8 | `seg_topic` |
| `/scovox/depth/image_raw` | `sensor_msgs/Image` (re-framed) | `depth_topic` |
| `/scovox/depth/camera_info` | `sensor_msgs/CameraInfo` (re-framed) | `depth_info_topic` |

Why re-publish depth: (1) it binds depth↔seg to one timestamp so SCovox's 0.05 s
sync always matches them regardless of inference lag; (2) SCovox's hardcoded
optical→body `kR` rotation needs a **body** depth `frame_id`, not the RealSense
optical frame. The node drops frames while the GPU is busy (natural throttle).

## Model

Default `facebook/mask2former-swin-large-mapillary-vistas-semantic` (Mapillary
Vistas, 65 outdoor classes) → collapsed to a compact 14-class outdoor set
(`outdoor_palette.py`, id 0 = other/unknown, 1..13 real). Weights are **CC-BY-NC**
(research use). Override with `-p model_name:=…`.

`outdoor_palette.py` is the single source of truth for the palette. To get the
matching SCovox params:

```bash
docker compose exec seg python3 -m seg_pipeline.outdoor_palette
# prints num_classes / max_semantic_classes / semantic_color_map_keys / _classes
```

Set those on the SCovox node (`num_classes: 14`, `max_semantic_classes: 14`, and the
`semantic_color_map_keys/classes`) so it decodes our colors back to the same ids.

## Run

```bash
# 1) build + start + launch the node (first build downloads torch cu128; first
#    run downloads the model weights into the `hf_cache` docker volume)
./run_seg.sh                 # use_sim_time:=true (bag playback)

# 2) in other containers: play the bag (--clock), and run the LiDAR localizer
#    publishing map -> ... -> base_link (NOT map->os_lidar). The bag's tf_static
#    chains base_link -> camera_color_frame, so the re-framed images resolve in map.

# 3) run the SCovox RGB-D node pointed at the three out topics above.
```

Useful params (`-p name:=value`): `depth_topic`, `color_topic`, `info_topic`,
`out_seg_topic`, `out_depth_topic`, `out_info_topic`, `output_frame`
(default `camera_color_frame`), `model_name`, `infer_short_edge` (default 720,
bounds VRAM/latency), `sync_slop` (default 0.05), `fp16` (default true).

## Requirements / notes

- GPU: built for Blackwell (RTX 50-series) via cu128 torch wheels; needs
  `nvidia-container-toolkit` on the host.
- The node logs the resolved model→compact class mapping at startup — **sanity-check
  it** (warns if the model's `id2label` looks wrong, e.g. ImageNet labels).
- This is the **one-hot colored-seg** path (argmax → palette). SCovox's richer
  per-pixel soft-label (`.topk`) path is file-based, not wired here.
