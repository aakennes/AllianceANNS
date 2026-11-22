#!/usr/bin/env bash
# AllianceANN shuffle/cache; read configs/<dataset>/shuffle.json.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/run_dataset.py" shuffle --entry-algorithm allianceann "$@"
