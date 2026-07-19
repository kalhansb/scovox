#!/bin/bash
# Run Experiment 1 at 10cm and 20cm resolutions, then compute metrics for all three.
# 5cm data already exists — only logs were regenerated.
#
# Usage: bash scripts/replica_run_all_resolutions.sh

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================"
echo "  Experiment 1: Multi-resolution runs"
echo "  10cm + 20cm data collection + all metrics"
echo "  Started: $(date)"
echo "========================================"
echo ""

# ── 10cm ──────────────────────────────────────────────
echo ">>> [1/5] Creating 10cm dataset..."
bash scripts/replica_create_dataset.sh 0.10
echo ""
echo ">>> [1/5] 10cm dataset DONE at $(date)"
echo ""

# ── 20cm ──────────────────────────────────────────────
echo ">>> [2/5] Creating 20cm dataset..."
bash scripts/replica_create_dataset.sh 0.20
echo ""
echo ">>> [2/5] 20cm dataset DONE at $(date)"
echo ""

# ── Metrics ───────────────────────────────────────────
METRICS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/results"

echo ">>> [3/5] Computing 5cm metrics..."
bash scripts/replica_compute_metrics.sh 0.05 | tee "$METRICS_DIR/metrics_5cm.txt"
echo ""

echo ">>> [4/5] Computing 10cm metrics..."
bash scripts/replica_compute_metrics.sh 0.10 | tee "$METRICS_DIR/metrics_10cm.txt"
echo ""

echo ">>> [5/5] Computing 20cm metrics..."
bash scripts/replica_compute_metrics.sh 0.20 | tee "$METRICS_DIR/metrics_20cm.txt"
echo ""

echo "========================================"
echo "  ALL DONE at $(date)"
echo "  Metrics saved to:"
echo "    $METRICS_DIR/metrics_5cm.txt"
echo "    $METRICS_DIR/metrics_10cm.txt"
echo "    $METRICS_DIR/metrics_20cm.txt"
echo "========================================"
