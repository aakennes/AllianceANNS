#!/usr/bin/env python3
"""Prepare mode-separated, content-checked static sidecars before search.

Only range predicates and structural routing metadata are used for automatic
landing statistics. No query vectors, ground truth or search results are read.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time

SCHEMA = 1


def boolean(value):
    if value not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path, full=True):
    path = Path(path).resolve(strict=True)
    st = path.stat()
    if not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    result = dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns)
    if full:
        result["sha256"] = digest(path)
    else:
        # Large immutable graph/layout identity: inode/stat plus header digest.
        with path.open("rb") as stream:
            result["header_sha256"] = hashlib.sha256(stream.read(4096)).hexdigest()
        result.update(device=st.st_dev, inode=st.st_ino, ctime_ns=st.st_ctime_ns)
    return result


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def histogram(path):
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"truncated landing file: {path}")
    count, = struct.unpack_from("<Q", data)
    if count == 0 or count > 100000 or len(data) != 8 * (count + 1):
        raise ValueError(f"invalid landing file shape: {path}")
    return list(struct.unpack_from(f"<{count}Q", data, 8))


def layout_can_prefilter(path):
    with Path(path).open("rb") as stream:
        header = stream.read(96)
    if len(header) != 96:
        raise ValueError("truncated ML layout header")
    fields = struct.unpack("<12Q", header)
    if fields[0] != 0x314D4C4C or fields[1] not in (1, 2):
        raise ValueError("unsupported ML layout header")
    return bool(fields[10])


@contextlib.contextmanager
def lock(path):
    import fcntl  # Linux server; no third-party dependencies.
    with Path(path).open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def valid_cache(directory, expected_key):
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["key"] != expected_key or not manifest["files"]:
            return None
        for name, sha in manifest["files"].items():
            if Path(name).name != name or digest(directory / name) != sha:
                return None
        return manifest
    except (OSError, ValueError, KeyError):
        return None


def publish(root, category, cache_key, build):
    directory = root / category / cache_key
    directory.parent.mkdir(parents=True, exist_ok=True)
    with lock(directory.parent / (cache_key + ".lock")):
        manifest = valid_cache(directory, cache_key)
        if manifest:
            return directory, manifest
        staging = Path(tempfile.mkdtemp(prefix=cache_key[:12] + ".pending-", dir=directory.parent))
        try:
            manifest = build(staging)
            manifest.update(schema_version=SCHEMA, key=cache_key)
            manifest["files"] = {p.name: digest(p) for p in staging.iterdir() if p.is_file()}
            write_json(staging / "manifest.json", manifest)
            if directory.exists():
                # Preserve damaged/stale managed cache and its logs for inspection.
                directory.rename(directory.with_name(cache_key + f".invalid-{time.time_ns()}"))
            staging.rename(directory)
            return directory, manifest
        finally:
            if staging.exists():
                # Keep validation failures and partial outputs as diagnostics.
                staging.rename(staging.with_name(staging.name.replace(".pending-", ".failed-")))


def execute(command, directory):
    write_json(directory / "command.json", command)
    with (directory / "prepare.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        # Keep the failed attempt under its own name; do not publish it as valid.
        failure = directory.with_name(directory.name.replace(".pending-", ".failed-"))
        directory.rename(failure)
        raise RuntimeError(f"preparation exit {result.returncode}; see {failure / 'prepare.log'}")
    return (directory / "prepare.log").read_text()


def prepare(args):
    budgets = sorted(set(int(x) for x in args.budgets.split(",")))
    if not budgets or any(x <= 0 for x in budgets):
        raise ValueError("budgets must contain positive byte counts")
    if args.print_sidecar and len(budgets) != 1:
        raise ValueError("print_sidecar requires a single budget")
    if args.query_limit < 0 or args.prefilter_threshold < 0:
        raise ValueError("query_limit and prefilter_threshold must be nonnegative")
    if args.static_cache_use_landing and not (args.static_landing_file or args.static_landing_query_rng or args.query_rng):
        raise ValueError("weighted mode needs query_rng, static_landing_query_rng or static_landing_file")
    if args.dry_run:
        return dict(dry_run=True, mode="landing" if args.static_cache_use_landing else "equal", budgets=budgets)
    root = Path(args.cache_dir).resolve()
    prepare_binary = Path(args.bin_dir) / "alliance_prepare_static_cache"
    source = dict(graph=identity(args.graph_file, False), meta=identity(args.meta_file), space=args.space)
    layout = dict(graph=identity(args.ml_graph_file, False), sidecar=identity(args.ml_meta_file))
    can_prefilter = layout_can_prefilter(args.ml_graph_file)
    profile = None
    landing = None
    routes = None
    counts = None
    if args.static_cache_use_landing:
        if args.static_landing_file:
            # Explicit externally trained/combined histogram, copied and hashed.
            external = Path(args.static_landing_file).resolve(strict=True)
            counts = histogram(external)
            profile = dict(schema_version=SCHEMA, source=source, external=identity(external))

            def build_hist(stage):
                shutil.copyfile(external, stage / "landing.bin")
                return dict(profile=profile, histogram=counts, origin="external", routing=None)
        else:
            ranges = Path(args.static_landing_query_rng or args.query_rng).resolve(strict=True)
            collector = Path(args.bin_dir) / "alliance_collect_landing"
            profile = dict(schema_version=SCHEMA, source=source, ranges=identity(ranges),
                           collector=identity(collector), filter_mode=args.filter_mode,
                           prefilter_threshold=args.prefilter_threshold, query_limit=args.query_limit,
                           can_prefilter=can_prefilter)

            def build_hist(stage):
                command = [str(collector), "--graph_file", args.graph_file, "--meta_file", args.meta_file,
                           "--query_rng", str(ranges), "--out_file", str(stage / "landing.bin"),
                           "--report_file", str(stage / "routing.json"), "--space", args.space,
                           "--filter_mode", args.filter_mode, "--prefilter_threshold", str(args.prefilter_threshold),
                           "--query_limit", str(args.query_limit), "--can_prefilter", str(can_prefilter).lower()]
                execute(command, stage)
                values = histogram(stage / "landing.bin")
                routing = json.loads((stage / "routing.json").read_text())
                if routing["histogram"] != values or sum(values) != routing["graph_queries"]:
                    raise ValueError("collector report does not match histogram")
                return dict(profile=profile, histogram=values, origin="range_structure", routing=routing)
        hist_dir, hist_manifest = publish(root, "landing", key(profile), build_hist)
        landing = hist_dir / "landing.bin"
        counts = hist_manifest["histogram"]
        routes = hist_manifest["routing"]
    allocation = "equal" if not args.static_cache_use_landing else ("landing" if sum(counts) else "equal_no_graph_landings")
    if allocation == "equal_no_graph_landings":
        print("WARN: no graph landings; static allocator falls back to equal level budgets", file=sys.stderr)
    side_profile = dict(schema_version=SCHEMA, source=source, layout=layout, budgets=budgets,
                        requested_mode="landing" if args.static_cache_use_landing else "equal",
                        allocation=allocation, landing_key=key(profile) if profile else None,
                        preparer=identity(prepare_binary))

    def build_sidecar(stage):
        command = [str(prepare_binary), "--graph_file", args.graph_file, "--meta_file", args.meta_file,
                   "--ml_graph_file", args.ml_graph_file, "--ml_meta_file", args.ml_meta_file,
                   "--space", args.space, "--budgets", ",".join(map(str, budgets)), "--out_prefix", str(stage / "static")]
        if landing:
            command += ["--static_landing_file", str(landing)]
        output = execute(command, stage)
        records = []
        for match in re.finditer(r"# static_prepared: path=(.*?) budget_bytes=(\d+) resident_bytes=(\d+) payload_bytes=(\d+) cached_nodes=(\d+)", output):
            path, budget, resident, payload, nodes = match.groups()
            record = dict(file=Path(path).name, budget_bytes=int(budget), resident_bytes=int(resident),
                          payload_bytes=int(payload), cached_nodes=int(nodes))
            if record["resident_bytes"] > record["budget_bytes"]:
                raise ValueError("prepared static resident bytes exceed budget")
            records.append(record)
        if sorted(r["budget_bytes"] for r in records) != budgets:
            raise ValueError("preparer output does not cover requested budgets")
        for record in records:
            if not (stage / record["file"]).is_file():
                raise ValueError("prepared sidecar missing")
        return dict(profile=side_profile, allocation=allocation, routing=routes, histogram=counts, outputs=records)

    side_dir, manifest = publish(root, side_profile["requested_mode"], key(side_profile), build_sidecar)
    return dict(mode=allocation, histogram=str(landing) if landing else None, routing=routes,
                manifest=str(side_dir / "manifest.json"),
                sidecars={str(r["budget_bytes"]): str(side_dir / r["file"]) for r in manifest["outputs"]})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("graph_file", "meta_file", "ml_graph_file", "ml_meta_file", "cache_dir", "budgets"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--bin_dir", default=os.environ.get("ALLIANCE_BIN_DIR", str(Path(__file__).resolve().parents[1] / "build/bin")))
    p.add_argument("--space", choices=("ip", "l2"), default="l2")
    p.add_argument("--static_cache_use_landing", type=boolean, default=False)
    p.add_argument("--query_rng")
    p.add_argument("--static_landing_query_rng")
    p.add_argument("--static_landing_file")
    p.add_argument("--filter_mode", choices=("auto", "graph", "prefilter", "legacy"), default="auto")
    p.add_argument("--prefilter_threshold", type=int, default=8192)
    p.add_argument("--query_limit", type=int, default=0)
    p.add_argument("--print_sidecar", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p


def main():
    args = parser().parse_args()
    try:
        result = prepare(args)
        if args.print_sidecar and not args.dry_run:
            print(next(iter(result["sidecars"].values())))
        else:
            print(json.dumps(result, indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
