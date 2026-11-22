#!/usr/bin/env bash
# AllianceANN build; read configs/<dataset>/build.json.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/run_dataset.py" build --entry-algorithm allianceann "$@"
