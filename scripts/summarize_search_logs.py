#!/usr/bin/env python3
"""Merge alliance_search_disk logs into one table.

disk_search_common.sh writes one log per (method, selectivity, bits, efs, R)
point, so a single sweep leaves dozens of files that have to be read by hand.
This script collapses them: one CSV row per log, plus a stdout table grouped by
"everything except efs" so an efs sweep reads as one block of lines.

Configuration is taken from the sibling <log>.command file (the full argv that
produced the log), not from the file name -- the name only encodes p/b/ef/R,
while --rerank_beam_width, --rerank_batch_blocks and --rerank_error_bound are
invisible in it.

Usage:
  python3 scripts/summarize_search_logs.py RUN_DIR [RUN_DIR ...] [-o summary.csv]
    --no-median     list every replicate instead of aggregating them
    --sort KEY      sort rows inside a group by KEY (default: efs)
    --columns A,B   override the displayed metric columns
    --quiet         write the CSV only
"""

import argparse
import csv
import os
import re
import shlex
import statistics
import sys

# Paths are recorded but never used to group: two runs of the same experiment
# from different checkouts must land in the same group.
PATH_KEYS = {
    "graph_file", "meta_file", "query_vec", "gt_file", "ml_graph_file",
    "ml_meta_file", "rabitq_centroids", "rabitq_cids", "query_rng",
}
# argv flags that take no value.
BOOL_FLAGS = {"per_query_cache", "rerank_error_bound", "exact", "memory"}

KV_RE = re.compile(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=([^\s,{}\[\]]*)")
COMMENT_RE = re.compile(r"^#\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")
REQUANT_RE = re.compile(r"^RequantizeRaBitQ:.*\(([0-9.e+-]+)s\)")

DEFAULT_COLUMNS = [
    "recall", "qps", "latency_avg_ms", "latency_p95_ms", "dist_comps",
    "ml_block_reads", "post_rerank_io", "ml_static_hit_rate",
    "memory_tunable_total_mib", "timing_io_pct", "timing_search_io_ms",
    "timing_rerank_io_ms", "timing_rerank_cpu_ms", "timing_cpu_ms",
]
SHORT = {
    "latency_avg_ms": "lat_avg", "latency_p95_ms": "lat_p95",
    "ml_block_reads": "blk_reads", "post_rerank_io": "rr_io",
    "timing_io_pct": "io%", "timing_search_io_ms": "s_io_ms",
    "timing_rerank_io_ms": "r_io_ms", "timing_rerank_cpu_ms": "r_cpu_ms",
    "timing_cpu_ms": "cpu_ms", "timing_search_cpu_ms": "s_cpu_ms",
    "timing_wall_ms": "wall_ms", "timing_setup_ms": "setup_ms",
    "dist_comps": "dist", "rerank_candidates": "rr_cand",
    "ml_static_hit_rate": "static%", "memory_tunable_total_mib": "mem_MiB",
    "ml_cache_resident_bytes_per_thread": "dyn_B",
    "ml_cache_resident_bytes_total": "dyn_total_B",
    "worker_memory_retained_total_bytes": "worker_B",
    "worker_memory_process_peak_rss_kib": "peak_RSS_KiB",
    "ml_cache_peak_slots": "pk_slots",
}


def parse_command(path):
    """Return the configuration dict encoded in a <log>.command file."""
    cfg = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            argv = shlex.split(fh.read())
    except OSError:
        return cfg
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "taskset" and i + 2 < len(argv) and argv[i + 1] == "-c":
            cfg["cpu"] = argv[i + 2]
            i += 3
            continue
        if tok.startswith("--"):
            key = tok[2:]
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if key in BOOL_FLAGS or nxt is None or nxt.startswith("--"):
                cfg[key] = "1"
                i += 1
            else:
                cfg[key] = nxt
                i += 2
            continue
        i += 1
    for key in BOOL_FLAGS:
        cfg.setdefault(key, "0")
    rng = cfg.get("query_rng", "")
    if rng:
        # sift_query_range_2-1.ibin -> 2-1 ; anything else keeps its basename.
        base = os.path.basename(rng)
        m = re.search(r"range[_-]([0-9]+-[0-9]+)", base)
        cfg["sel"] = m.group(1) if m else base
    return cfg


def parse_log(path):
    """Return the metrics dict of one search log, or None if it has no results."""
    metrics = {}
    csv_header = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            m = COMMENT_RE.match(line)
            if m:
                name, payload = m.group(1), m.group(2)
                for k, v in KV_RE.findall(payload):
                    metrics[f"{name}_{k}"] = v
                continue
            if line.startswith("#"):
                continue
            if line.startswith("efs,"):
                csv_header = line.split(",")
                continue
            if csv_header and re.match(r"^[0-9]", line):
                row = line.split(",")
                if len(row) == len(csv_header):
                    metrics.update(zip(csv_header, row))
                    csv_header = None
                continue
            m = REQUANT_RE.match(line)
            if m:
                metrics["requant_s"] = m.group(1)
                continue
            if line.startswith("InitRaBitQ:"):
                for k, v in KV_RE.findall(line):
                    metrics[k] = v
                continue
            if line.lstrip().startswith("k=") and "beam_width=" in line:
                for k, v in KV_RE.findall(line):
                    metrics[f"hdr_{k}"] = v
    return metrics if "recall" in metrics else None


def collect(targets):
    rows = []
    skipped = []
    logs = []
    for target in targets:
        if os.path.isfile(target):
            logs.append(target)
            continue
        for root, _dirs, files in os.walk(target):
            for name in sorted(files):
                if name.endswith(".log"):
                    logs.append(os.path.join(root, name))
    for log in sorted(logs):
        metrics = parse_log(log)
        if metrics is None:
            skipped.append(log)
            continue
        cfg = parse_command(log + ".command")
        row = {"log": log}
        row.update({f"cfg_{k}": v for k, v in cfg.items()})
        row.update(metrics)
        row["_cfg"] = cfg
        rows.append(row)
    return rows, skipped


def group_keys(rows):
    """Config keys that define a group: everything except efs and paths.

    A key whose value is a function of efs across the whole input (the
    `--rerank_num heuristic` case, where R scales with efs) is dropped, so an
    efs sweep still collapses into a single group.
    """
    keys = set()
    for row in rows:
        keys.update(row["_cfg"].keys())
    keys -= PATH_KEYS
    keys.discard("efs")
    efs_values = {row["_cfg"].get("efs") for row in rows}
    if len(efs_values) >= 2:
        for key in sorted(keys):
            values = {row["_cfg"].get(key) for row in rows}
            if len(values) < 2:
                continue
            by_efs = {}
            functional = True
            for row in rows:
                ef = row["_cfg"].get("efs")
                val = row["_cfg"].get(key)
                if by_efs.setdefault(ef, val) != val:
                    functional = False
                    break
            if functional:
                keys.discard(key)
    return sorted(keys)


def fmt(value):
    if value in (None, ""):
        return "-"
    try:
        num = float(value)
    except ValueError:
        return str(value)
    if num == int(num) and abs(num) < 1e15:
        return str(int(num))
    if abs(num) >= 100:
        return f"{num:.1f}"
    if abs(num) >= 1:
        return f"{num:.3f}"
    return f"{num:.6f}"


def median_of(values):
    nums = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            pass
    return statistics.median(nums) if nums else None


def print_table(rows, keys, columns, sort_key, use_median):
    groups = {}
    for row in rows:
        gk = tuple(row["_cfg"].get(k, "") for k in keys)
        groups.setdefault(gk, []).append(row)
    # Only the config keys that actually differ between groups go in the label.
    varying = [k for i, k in enumerate(keys)
               if len({gk[i] for gk in groups}) > 1] or keys

    for gk in sorted(groups):
        label = " ".join(f"{k}={gk[keys.index(k)]}" for k in varying)
        members = groups[gk]
        print(f"\n== {label or 'all'}  ({len(members)} log(s))")
        present = [c for c in columns if any(row.get(c) not in (None, "") for row in members)]
        buckets = {}
        for row in members:
            buckets.setdefault(row.get("efs", ""), []).append(row)

        def ef_sort(item):
            try:
                return (0, float(item))
            except (TypeError, ValueError):
                return (1, 0.0)

        head = ["efs", "n"] + [SHORT.get(c, c) for c in present]
        table = [head]
        for ef in sorted(buckets, key=ef_sort):
            reps = buckets[ef]
            if use_median:
                line = [str(ef), str(len(reps))]
                for col in present:
                    med = median_of(r.get(col) for r in reps)
                    cell = fmt(med) if med is not None else fmt(reps[0].get(col))
                    if len(reps) > 1 and med is not None:
                        vals = [v for v in (median_of([r.get(col)]) for r in reps) if v is not None]
                        if vals and max(vals) != min(vals) and med:
                            spread = (max(vals) - min(vals)) / abs(med) * 100
                            if spread >= 0.05:
                                cell += f"[{spread:.1f}%]" if spread < 10 else f"[{spread:.0f}%]"
                    line.append(cell)
                table.append(line)
            else:
                for rep in sorted(reps, key=lambda r: r["log"]):
                    table.append([str(ef), "1"] + [fmt(rep.get(c)) for c in present])
        widths = [max(len(r[i]) for r in table) for i in range(len(head))]
        for i, line in enumerate(table):
            print("  " + "  ".join(cell.rjust(widths[j]) for j, cell in enumerate(line)))
            if i == 0:
                print("  " + "  ".join("-" * w for w in widths))
    if use_median:
        print("\n[x%] = (max-min)/median spread across replicates.")


def main():
    ap = argparse.ArgumentParser(description="Summarize alliance_search_disk logs.")
    ap.add_argument("targets", nargs="+", help="run directories or individual logs")
    ap.add_argument("-o", "--out", help="write summary.csv here")
    ap.add_argument("--no-median", action="store_true",
                    help="list every replicate instead of taking the median")
    ap.add_argument("--sort", default="efs", help="unused placeholder; rows sort by efs")
    ap.add_argument("--columns", help="comma-separated metric columns to display")
    ap.add_argument("--quiet", action="store_true", help="write the CSV, print no table")
    args = ap.parse_args()

    rows, skipped = collect(args.targets)
    if not rows:
        print("no search logs with results found", file=sys.stderr)
        return 1

    keys = group_keys(rows)
    columns = args.columns.split(",") if args.columns else DEFAULT_COLUMNS

    if args.out:
        fields = []
        for row in rows:
            for k in row:
                if k != "_cfg" and k not in fields:
                    fields.append(k)
        fields = ["log"] + sorted(f for f in fields if f != "log")
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: v for k, v in row.items() if k != "_cfg"})
        print(f"wrote {args.out} ({len(rows)} rows, {len(fields)} columns)")

    if not args.quiet:
        print_table(rows, keys, columns, args.sort, not args.no_median)
    if skipped:
        print(f"\nskipped {len(skipped)} file(s) with no result row "
              f"(e.g. {os.path.basename(skipped[0])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
