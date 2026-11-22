#!/usr/bin/env bash
set -euo pipefail
K=${K:-128}
DATA_ROOT="${DATA_ROOT:-${ALLIANCE_DATA_ROOT:-$HOME/data}/Sift1m}"
python3 ./scripts/python/ivf.py "$DATA_ROOT/sift_base.fvecs" \
  "$K" "$DATA_ROOT/Centroids/base.1M.centroids${K}.fvecs" \
  "$DATA_ROOT/Centroids/base.1M.cids${K}.ivecs" l2
