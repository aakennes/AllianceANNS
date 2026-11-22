#!/usr/bin/env python3
"""Validate frozen CaseLaw configuration metadata and print a read-only plan."""

import argparse
import json
from pathlib import Path
import sys

# A metadata-only CLI must not create import caches in the source tree either.
sys.dont_write_bytecode = True
from benchlib.protocol import load_plans, read_json, WORKLOADS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/caselaw/profiles.json")
    parser.add_argument("--workload", choices=WORKLOADS, action="append", help="Repeat to select workloads; default: all five")
    parser.add_argument("--paths", type=Path, help="JSON object overriding path_roots for display; does not relocate frozen drivers")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and print the protocol/plan (default); no files are written")
    mode.add_argument("--validate", action="store_true", help="Same read-only validation and plan, including bundled evidence hashes")
    args = parser.parse_args(argv)
    try:
        overrides = read_json(args.paths) if args.paths else None
        plans = load_plans(args.config, args.workload, overrides)
        print(json.dumps({"planner": "caselaw", "plans": plans}, indent=2, ensure_ascii=False))
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        parser.exit(2, "ERROR: " + str(error) + "\n")


if __name__ == "__main__":
    main()
