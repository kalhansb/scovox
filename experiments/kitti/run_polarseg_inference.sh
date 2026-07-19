#!/bin/bash
# Generate PolarNet predictions (raw-ID .label) for KITTI seqs 06-10 into
# <dataset>/sequences/<seq>/predictions/. Runs under the `polarnet` conda env
# (torch 2.1+cu118, Pascal sm_61 OK) from the PolarSeg repo dir.
set -eo pipefail
WS=/home/kalhan/Projects/scovox_ws
CONDA=/home/kalhan/miniconda3/bin/conda
PSDIR="${WS}/third_party/PolarSeg"
DATASET="${WS}/data/semantickitti/dataset"
SEQS="${1:-06,07,08,09,10}"
DEVICE="${2:-cuda:0}"

cd "${PSDIR}"
"${CONDA}" run -n polarnet python run_polarseg_scovox.py \
    --dataset "${DATASET}" --seqs "${SEQS}" --n_scans 100 --device "${DEVICE}"
