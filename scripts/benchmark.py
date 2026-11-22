#!/usr/bin/env python3
"""Plan, preflight, execute, or resume a fingerprinted benchmark job."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from benchlib.datasets import dispatch as build_experiment
from benchlib.spec import fingerprint_inputs, validate_spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parent
    parser.add_argument("--config", type=Path, default=directory / "configs/caselaw/profiles.json")
    parser.add_argument("--bindings", type=Path, default=directory / "configs/caselaw/runtime.json")
    parser.add_argument("--workload", choices=("origin", "mixed", "p1", "p4", "p7", "p10"), action="append")
    parser.add_argument("--algorithm", choices=("allianceann", "pipewow", "pipeannfilter"), action="append")
    parser.add_argument("--cache-policy", choices=("landing", "equal", "native", "off"), action="append")
    parser.add_argument("--job", type=Path, help="New output directory for --run, existing directory for --resume")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Print commands and bindings without opening inputs (default)")
    mode.add_argument("--validate", action="store_true", help="Verify asset availability and full SHA256; no queries or files written")
    mode.add_argument("--run", action="store_true", help="Execute a new job using its pinned native drivers")
    mode.add_argument("--resume", action="store_true", help="Resume verification/export and unsent curves; never replay a sent measurement")
    args = parser.parse_args(argv)
    if (args.run or args.resume) != (args.job is not None):
        parser.error("--run/--resume requires --job; planning/preflight does not accept --job")
    try:
        spec = build_experiment(args.config, args.bindings, args.workload, args.algorithm, args.cache_policy)
        if args.validate:
            identities = fingerprint_inputs(spec)
            result = {"status": "inputs_verified", "formal_queries_run": 0, "inputs": identities, "spec": spec}
        elif args.run or args.resume:
            from benchlib.runner import run_experiment
            validate_spec(spec)
            result = run_experiment(spec, args.job, resume=args.resume)
        else:
            result = {"status": "planned", "production_inputs_checked": False, "formal_queries_run": 0, "spec": spec}
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
