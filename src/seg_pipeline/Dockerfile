# ---------------------------------------------------------------------------
# Online RGB-D segmentation container (ROS 2 Jazzy + PyTorch/CUDA + transformers).
#
# Runs the Mask2Former-Mapillary segmenter as a live ROS node, sharing the DDS
# graph (host net + ROS_DOMAIN_ID) with the scovox + localizer containers.
#
# GPU: built for Blackwell (RTX 50-series, sm_120) via the cu128 torch wheels.
# Needs nvidia-container-toolkit on the host (same as the GLIM CUDA container).
# ---------------------------------------------------------------------------
FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        ros-jazzy-message-filters \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA 12.8 (Blackwell support). Ubuntu 24.04 python is
# externally-managed, so --break-system-packages installs alongside apt rclpy.
RUN pip3 install --no-cache-dir --break-system-packages \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision

# HF stack + image IO (Pillow only — no OpenCV, so the Debian numpy 1.26 the ROS
# stack is built against stays put; we decode/resize with Pillow + numpy).
RUN pip3 install --no-cache-dir --break-system-packages \
        "transformers>=4.43" \
        "huggingface_hub>=0.23" \
        safetensors \
        pillow

# Persisted HF cache (mounted as a volume in compose).
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /seg
COPY outdoor_palette.py seg_node.py /seg/

# Source ROS for every shell.
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc

CMD ["bash"]
