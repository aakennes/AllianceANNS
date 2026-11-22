"""Export only independently validated measurements with paired-repeat decisions."""

import csv
import io
import json
from pathlib import Path

from .metrics import aggregate_repeats
from .spec import digest_file, require, validate_spec


METRICS = ("recall", "qps", "p95_ms", "io_requests_per_query", "io_4k_equivalent_per_query",
           "io_8k_equivalent_per_query", "io_bytes_per_query", "batch_wall_seconds")
OPTIONAL_METRICS = ("graph_reads_4k_per_query", "vector_reads_8k_per_query", "static_hits_per_query")


def write_atomic(path, content):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        stream.write(content)
    tmp.replace(path)


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def csv_text(rows):
    keys = sorted({key for row in rows for key in row})
    first = [key for key in ("curve_id", "algorithm", "workload", "cache_policy", "bits", "ef", "repeat", "queries") if key in keys]
    fields = first + [key for key in keys if key not in first]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def finalize_experiment(spec, job_dir, rows):
    """No native process is launched here. Every expected repeat must be present."""
    validate_spec(spec)
    rows = [dict(row) for row in rows]
    curves = {curve["id"]: curve for curve in spec["curves"]}
    expected = {(curve["id"], ef, repeat) for curve in curves.values()
                for ef in curve["efs"] for repeat in range(1, curve["repeats"] + 1)}
    actual = [(row.get("curve_id"), row.get("ef"), row.get("repeat")) for row in rows]
    require(len(actual) == len(set(actual)) and set(actual) == expected, "incomplete or duplicate measurement coverage")
    for row in rows:
        curve = curves[row["curve_id"]]
        require(row.get("validated") is True, "unvalidated repeat cannot be exported")
        for field in ("algorithm", "workload", "cache_policy", "bits", "queries"):
            require(row.get(field) == curve[field], "measurement identity changed: " + field)
        require(row.get("search_peak_rss_bytes", -1) >= 0 and row["search_peak_rss_bytes"] <= spec["memory"]["search_rss_limit_bytes"],
                "search RSS acceptance failed")
        require(row.get("loading_peak_rss_bytes", -1) >= 0 and row["loading_peak_rss_bytes"] <= spec["memory"]["loading_limit_bytes"],
                "loading RSS acceptance failed")
    rows.sort(key=lambda row: (row["curve_id"], row["ef"], row["repeat"]))
    summary, decisions = [], []
    for cid, curve in curves.items():
        for ef in curve["efs"]:
            group = [row for row in rows if row["curve_id"] == cid and row["ef"] == ef]
            optional = []
            for metric in OPTIONAL_METRICS:
                present = [metric in row for row in group]
                require(all(present) or not any(present), "inconsistent optional metric coverage")
                if all(present):
                    optional.append(metric)
            result = aggregate_repeats(group, METRICS + tuple(optional), spec["aggregation"]["policy"], curve["repeats"])
            identity = {key: curve[key] for key in ("algorithm", "workload", "cache_policy", "bits", "queries")}
            identity.update(curve_id=cid, ef=ef)
            summary.append(dict(identity, **result["metrics"], measured_repeats=result["measured_repeats"],
                                retained_repeats=result["retained_repeats"],
                                max_search_peak_rss_bytes=max(row["search_peak_rss_bytes"] for row in group),
                                max_loading_peak_rss_bytes=max(row["loading_peak_rss_bytes"] for row in group)))
            decisions.append(dict(identity, **result))
    job = Path(job_dir).resolve()
    out = job / "aggregate_export"
    require(not out.is_symlink() and out.resolve().parent == job, "export directory must remain inside its job")
    out.mkdir(exist_ok=True)
    artifacts = {"summary.csv": csv_text(summary), "trim_decisions.json": json_text(decisions),
                 "request.json": json_text(spec)}
    for algorithm in sorted({row["algorithm"] for row in rows}):
        artifacts[algorithm + "_raw.csv"] = csv_text([row for row in rows if row["algorithm"] == algorithm])
    verification = {"status": "verified", "measured_batches": len(rows), "aggregate_points": len(summary),
                    "queries": sum(row["queries"] for row in rows), "all_repeats_validated_before_trimming": True,
                    "policy": spec["aggregation"]["policy"], "per_query_results_exported": False,
                    "timing": ("Native batch timing includes observer; GT checks, loading, raw export and replay are outside timing."
                               if spec.get("native_protocol") == "alliance-formal-v1" else
                               "Native batch timing includes observer and in-batch GT checks; loading, raw export and independent replay are outside timing."),
                    "io": "Actual native requests are retained separately; 4/8 KiB equivalents are requested bytes divided by 4096/8192.",
                    "p95": "Mean of retained batch P95 values, not a pooled-query percentile.",
                    "memory": spec["memory"]}
    artifacts["verification.json"] = json_text(verification)
    allowed = set(artifacts) | {"export_manifest.json", "export_manifest.json.tmp"} | {name + ".tmp" for name in artifacts}
    require(not any(p.name not in allowed or p.is_symlink() or p.is_dir() for p in out.iterdir()),
            "unexpected aggregate export content")
    for name, text in artifacts.items():
        write_atomic(out / name, text)
    manifest = {name: digest_file(out / name) for name in sorted(artifacts)}
    write_atomic(out / "export_manifest.json", json_text(manifest))
    return {"status": "verified", "export_directory": str(out), "aggregate_points": len(summary),
            "measured_batches": len(rows), "queries": verification["queries"], "files": manifest}
