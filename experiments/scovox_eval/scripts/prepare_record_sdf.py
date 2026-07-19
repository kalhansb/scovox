#!/usr/bin/env python3
"""Inject TrajectoryRecorder plugin + camera model into a world SDF.

Creates a recording-ready SDF by:
1. Copying the original world SDF
2. Inserting the trajectory_camera model
3. Inserting the TrajectoryRecorder plugin
4. Setting real_time_factor=0 for max speed

Usage:
    python prepare_record_sdf.py \
        --world src/hmr_sim/worlds/flatforest/flatforestv2.sdf \
        --trajectory results/flatforest_rendered/poses.json \
        --output-dir results/flatforest_rendered \
        --output-sdf /tmp/flatforest_record.sdf
"""

import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Inject TrajectoryRecorder into a world SDF"
    )
    parser.add_argument("--world", required=True, help="Original world SDF")
    parser.add_argument("--trajectory", required=True, help="Path to poses.json")
    parser.add_argument("--output-dir", required=True, help="Where to save frames")
    parser.add_argument("--output-sdf", required=True, help="Output SDF path")
    parser.add_argument("--settle-steps", type=int, default=20)
    args = parser.parse_args()

    traj_abs = str(Path(args.trajectory).resolve())
    outdir_abs = str(Path(args.output_dir).resolve())

    sdf = Path(args.world).read_text()

    # Set real_time_factor to 1 (real-time — needed for sensor rendering to keep up)
    sdf = re.sub(
        r"<real_time_factor>[^<]*</real_time_factor>",
        "<real_time_factor>1</real_time_factor>",
        sdf,
    )

    # Plugin + camera model block to inject
    inject = f"""
    <!-- === Injected by prepare_record_sdf.py === -->
    <plugin name="trajectory_recorder::TrajectoryRecorder" filename="TrajectoryRecorder">
      <trajectory_file>{traj_abs}</trajectory_file>
      <output_dir>{outdir_abs}</output_dir>
      <model_name>trajectory_camera</model_name>
      <settle_steps>{args.settle_steps}</settle_steps>
    </plugin>

    <include>
      <uri>model://trajectory_camera</uri>
      <name>trajectory_camera</name>
      <pose>0 0 1.5 0 0 0</pose>
    </include>
    <!-- === End injection === -->
"""

    # Insert before </world>
    sdf = sdf.replace("</world>", inject + "\n  </world>")

    Path(args.output_sdf).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_sdf).write_text(sdf)
    print(f"Created: {args.output_sdf}")
    print(f"  Trajectory: {traj_abs}")
    print(f"  Output dir: {outdir_abs}")


if __name__ == "__main__":
    main()
