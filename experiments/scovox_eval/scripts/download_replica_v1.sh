#!/usr/bin/env bash
# Download Replica v1 and extract ONLY the 8 scenes we use.
#
# Source: facebookresearch/Replica-Dataset release v1.0 (17-part tarball).
# The upstream download.sh extracts the entire ~100 GB dataset; this wrapper
# streams through the tarball and keeps only the scenes we need, then deletes
# the .part?? archives to reclaim disk.

set -e

DEST="${1:-$(pwd)/data/replica_v1}"
WORK="${DEST}.download"

SCENES=(room_0 room_1 room_2 office_0 office_1 office_2 office_3 office_4)

mkdir -p "$WORK"
mkdir -p "$DEST"
cd "$WORK"

echo "Downloading 17 tarball parts (~100 GB) to $WORK ..."
for p in {a..q}; do
    wget --continue "https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta$p"
done

# Build the list of tar match patterns. Replica tarball layout is
# <root>/<scene>/... so we match '*/<scene>/*' for each scene.
patterns=()
for s in "${SCENES[@]}"; do
    patterns+=("*/$s/*")
done

echo "Streaming extract of selected scenes into $DEST ..."
cat replica_v1_0.tar.gz.part?? | unpigz -p 8 | \
    tar -xv --wildcards -C "$DEST" "${patterns[@]}"

echo "Fetching additional habitat configs ..."
wget -q "http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip" -O /tmp/habitat_configs.zip
unzip -qn /tmp/habitat_configs.zip -d "$DEST"
rm -f /tmp/habitat_configs.zip

echo "Cleaning up .part archives ..."
rm -f replica_v1_0.tar.gz.part??
cd "$DEST"
rmdir "$WORK" 2>/dev/null || true

echo "Done. Extracted scenes:"
ls -d "$DEST"/*/ 2>/dev/null || true
