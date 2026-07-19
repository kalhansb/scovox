#!/bin/bash
# Step 2 of flatforest GT pipeline: run WorldQuerySystem in Gazebo to get
# per-voxel ground truth (occupancy + semantic labels) for calibration eval.
#
# Prerequisites:
#   - flatforest_create_dataset.sh completed (scovox.npz, covox.npz, logodds.npz exist)
#   - Workspace built: colcon build && source install/setup.bash
#
# Usage:
#   source install/setup.bash
#   bash src/robot_sw/distributed_mapping/scovox_eval/scripts/flatforest_gt_query.sh
#
# What this does:
#   1. Creates query_points_union.csv (union of all 3 variants' voxel positions)
#   2. Injects WorldQuerySystem plugin into flatforestv2.sdf
#   3. Runs Gazebo headless to query GT at all positions
#   4. Converts output CSV to gt_voxels.npz
#
# Output: results/flatforest_10cm/gt_voxels.npz
#   - points (N,3) float32
#   - gt_binary (N,) float32  — 1.0=occupied, 0.0=free
#   - semantic_class (N,) int32 — semantic label ID

set -e

WS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws"
EVAL_PKG="$WS/src/robot_sw/distributed_mapping/scovox_eval"
RESULTS="$EVAL_PKG/results/flatforest_10cm"
SRC_SDF="$WS/src/hmr_sim/worlds/flatforest/flatforestv2.sdf"
QUERY_CSV="$RESULTS/query_points_union.csv"
QUERY_SDF="/tmp/flatforest_gt_query.sdf"
OUTPUT_CSV="$RESULTS/world_query_results.csv"
GT_NPZ="$RESULTS/gt_voxels.npz"

# ── Step 1: Create union of query points (if not already done) ──────────
if [ -f "$QUERY_CSV" ]; then
    echo "Step 1: query_points_union.csv already exists ($(wc -l < "$QUERY_CSV") lines)"
else
    echo "Step 1: Creating union of query points from all 3 variants..."
    python3 -c "
import numpy as np, os, time
results = '$RESULTS'
pts_all = []
for name in ['scovox', 'covox', 'logodds']:
    d = np.load(os.path.join(results, f'{name}.npz'))
    p = d['points']
    print(f'  {name}: {len(p):,} voxels')
    pts_all.append(p)
all_pts = np.vstack(pts_all)
quantized = np.round(all_pts * 1000).astype(np.int64)
_, idx = np.unique(quantized, axis=0, return_index=True)
unique_pts = all_pts[idx]
print(f'  Unique positions: {len(unique_pts):,}')
with open('$QUERY_CSV', 'w') as f:
    f.write('x,y,z\n')
    for p in unique_pts:
        f.write(f'{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}\n')
print(f'  Wrote {len(unique_pts):,} points to $QUERY_CSV')
"
fi

# ── Step 2: Create query SDF ────────────────────────────────────────────
echo ""
echo "Step 2: Injecting WorldQuerySystem into flatforestv2.sdf..."
python3 -c "
with open('$SRC_SDF') as f:
    sdf = f.read()

plugin_xml = '''
    <!-- WorldQuerySystem: query GT occupancy + semantic labels -->
    <plugin name=\"world_query_system::WorldQuerySystem\" filename=\"WorldQuerySystem\">
      <csv_file>$QUERY_CSV</csv_file>
      <output_csv_file>$OUTPUT_CSV</output_csv_file>
      <ray_length>0.05</ray_length>
      <wait_iterations>100</wait_iterations>
    </plugin>'''

pos = sdf.find('<gravity>')
sdf_new = sdf[:pos] + plugin_xml + '\n    ' + sdf[pos:]

with open('$QUERY_SDF', 'w') as f:
    f.write(sdf_new)
print(f'  Wrote $QUERY_SDF')
"

# ── Step 3: Run Gazebo headless ─────────────────────────────────────────
echo ""
echo "Step 3: Running Gazebo headless (this may take 1-4 hours for ~18M points)..."
echo "  Query points: $(wc -l < "$QUERY_CSV") lines"
echo "  Output: $OUTPUT_CSV"
echo ""

# Ensure plugin and resource paths are set
export GZ_SIM_SYSTEM_PLUGIN_PATH="${WS}/install/hmr_sim/lib/hmr_sim${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export IGN_GAZEBO_RESOURCE_PATH="${WS}/src/hmr_sim/models:${WS}/install/hmr_sim/share/hmr_sim/models${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"

ign gazebo -s -v 3 "$QUERY_SDF"

# ── Step 4: Convert results to NPZ ─────────────────────────────────────
echo ""
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "ERROR: WorldQuerySystem did not produce $OUTPUT_CSV"
    exit 1
fi

echo "Step 4: Converting results to NPZ..."
python3 "$EVAL_PKG/scripts/gz_gt_to_npz.py" "$OUTPUT_CSV" "$GT_NPZ"

echo ""
echo "=== FLATFOREST GT QUERY COMPLETE ==="
echo "  GT file: $GT_NPZ"
echo ""
echo "Verify:"
echo "  python3 -c \"import numpy as np; d=np.load('$GT_NPZ'); print('Points:', d['points'].shape, 'Occupied:', int(d['gt_binary'].sum()), 'Labels:', sorted(set(d['semantic_class'])))\""
