# ---------------------------------------------------------------------------
# SCovox standalone build / test image.
#
# Self-contained ROS 2 Jazzy toolchain for the scovox workspace
# (scovox_msgs, scovox_core, scovox_mapping). This image is independent of the
# hmr_localisation image — scovox builds and tests on its own.
#
# All build + test dependencies are baked in via rosdep over the package
# manifests, so `colcon build` inside the container needs no network. Bonxai is
# vendored at scovox_core/include/third_party/bonxai (no external fetch); Eigen3,
# liblz4 and ament_cmake_gtest come from rosdep.
#
# Build the image (run from the repo root, i.e. the dir containing src/):
#   docker build -t scovox:jazzy -f docker/Dockerfile .
#
# Build + test the workspace (source bind-mounted; nothing baked into the image):
#   docker run --rm -v "$PWD":/scovox scovox:jazzy \
#       bash -lc 'colcon build && colcon test && colcon test-result --all'
#
# Interactive dev shell (ROS + workspace overlay auto-sourced):
#   docker run --rm -it -v "$PWD":/scovox scovox:jazzy
#
# Or just run docker/build_and_test.sh, which wraps both steps.
# ---------------------------------------------------------------------------
FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive

# Build tooling. The ROS-side deps (rclcpp, tf2*, cv_bridge, message_filters,
# sensor/geometry/nav_msgs, eigen3_cmake_module, liblz4-dev, ament_cmake_gtest)
# are resolved from the package.xml manifests by rosdep below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        python3-rosdep \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Resolve + install every declared dependency from the package manifests. Only
# the manifests are copied (not the sources), so this layer stays cached unless a
# package.xml changes; the actual sources are bind-mounted at run time.
COPY src/scovox_msgs/package.xml    /tmp/deps/src/scovox_msgs/package.xml
COPY src/scovox_core/package.xml    /tmp/deps/src/scovox_core/package.xml
COPY src/scovox_mapping/package.xml /tmp/deps/src/scovox_mapping/package.xml
RUN apt-get update \
    && rosdep update --rosdistro jazzy \
    && rosdep install --from-paths /tmp/deps/src --ignore-src -y --rosdistro jazzy \
    && rm -rf /tmp/deps /var/lib/apt/lists/*

# Auto-source ROS + the workspace overlay for every interactive shell.
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc && \
    echo '[ -f /scovox/install/setup.bash ] && source /scovox/install/setup.bash' >> /root/.bashrc

WORKDIR /scovox
CMD ["bash"]
