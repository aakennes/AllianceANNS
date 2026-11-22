#!/usr/bin/env bash
# AllianceANN search; read configs/<dataset>/search.json and efs.json.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/run_dataset.py" search --entry-algorithm allianceann "$@"
