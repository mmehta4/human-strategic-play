#!/usr/bin/env bash
# Fetch the 12-study dataset files from the ElementaryNet repository at a pinned commit.
# Skips the unused cn_cgc06 game files (~430 MB). Usage: scripts/fetch_wlb_data.sh [dest]
set -euo pipefail
DEST="${1:-$HOME/.cache/hsb/elementarynet}"
COMMIT=69005d68b367b8cab4a51327ae00061d2c545d90
if [ ! -d "$DEST/.git" ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/gregdeon/elementarynet.git "$DEST"
fi
git -C "$DEST" sparse-checkout set --no-cone '/*' '!/torch_bgt/data/cn_cgc06_*'
git -C "$DEST" checkout -q "$COMMIT"
echo "Data repo at $(git -C "$DEST" rev-parse HEAD) in $DEST"
