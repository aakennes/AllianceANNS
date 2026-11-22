"""Validate execution plans and freeze file identities without launching queries."""

import hashlib
from pathlib import Path
import re


ALGORITHMS = ("allianceann", "pipewow", "pipeannfilter")
SCOPE_PREFIXES = {"allianceann": "caselaw-rss4-r12-", "pipewow": "caselaw-pw-rss4-",
                  "pipeannfilter": "caselaw-paf-origin-rss3-"}
RETIRED_IN_TREE_SEARCH_OPTIONS = frozenset({
    "--memory", "--enum_thresh", "--export_threads", "--exact", "--basevec",
})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return type(value) is int and value > 0


def absolute(path):
    return isinstance(path, str) and bool(path) and "\x00" not in path and (
        path.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", path))


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stat(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


def validate_navigation_arguments(argv, by_path):
    """Require every selected RAM graph to have an explicit, pinned input path."""
    flags = {"--navigation_file", "--navigation_ef", "--navigation_budget", "--navigation_count"}
    route_flags = {"--entry_sampling", "--pipeline", "--filter_mode", "--ml_graph_file", "--ml_meta_file"}
    options, route = {}, {}
    for i, arg in enumerate(argv[1:], 1):
        key = arg.split("=", 1)[0]
        if key not in flags | route_flags:
            continue
        if key in route_flags:
            require(arg in route_flags and i + 1 < len(argv) and
                    not argv[i + 1].startswith("--"), "invalid navigation routing argument")
            route[arg] = argv[i + 1]  # The CLI uses the final repeated route option.
            continue
        require(arg == key and arg not in options and i + 1 < len(argv) and
                not argv[i + 1].startswith("--"), "invalid or duplicate navigation argument")
        options[arg] = argv[i + 1]
    pipeline = route.get("--pipeline", "0")
    require(re.fullmatch(r"[+-]?[0-9]+", pipeline) is not None, "invalid navigation pipeline argument")
    graph_route = (int(pipeline) != 0 and bool(route.get("--ml_graph_file") or route.get("--ml_meta_file")) and
                   route.get("--filter_mode", "legacy") in ("graph", "auto"))
    sampling = route.get("--entry_sampling")
    selected = sampling == "navigation" or (sampling is None and graph_route)
    if not options and not selected:
        return
    path = options.get("--navigation_file")
    require(path in by_path and by_path[path].get("sha256") is not None,
            "navigation requires an explicit --navigation_file with a pinned SHA256 in the input ledger")
    controls = {}
    for key, default in (("ef", 64), ("budget", 256), ("count", 16)):
        value = options.get("--navigation_" + key, str(default))
        require(re.fullmatch(r"[0-9]+", value) is not None, "invalid navigation integer")
        controls[key] = int(value)
    require(0 < controls["count"] <= min(controls["ef"], controls["budget"]) and
            controls["ef"] <= 65536 and controls["budget"] <= 65536,
            "navigation requires 0 < count <= min(ef, budget), ef/budget <= 65536")


def validate_spec(spec):
    """Check structure/protocol bindings only; file IO belongs to preflight."""
    require(isinstance(spec, dict) and spec.get("schema_version") in (2, 3), "unsupported execution schema")
    generic = spec["schema_version"] == 3
    if generic:
        require(spec.get("native_protocol") == "alliance-formal-v1" and spec.get("adapter") == "dataset-native-v1",
                "unsupported in-tree protocol")
    require(not spec.get("binding_errors"), "incomplete runtime bindings: " + str(spec.get("binding_errors", [])))
    require(isinstance(spec.get("id"), str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}", spec["id"]),
            "invalid experiment ID")
    memory = spec.get("memory", {})
    require(positive(memory.get("loading_limit_bytes")) and positive(memory.get("search_rss_limit_bytes"))
            and memory["search_rss_limit_bytes"] <= memory["loading_limit_bytes"], "invalid phase memory limits")
    aggregation = spec.get("aggregation", {})
    require((aggregation.get("policy") == "qps-paired-trim" and aggregation.get("repeats") == 5) or
            (generic and aggregation.get("policy") == "mean" and positive(aggregation.get("repeats"))),
            "execution requires five valid repeats with QPS-paired trim")
    assets = spec.get("inputs")
    require(isinstance(assets, list) and assets, "execution requires an input identity ledger")
    by_path = {}
    for asset in assets:
        require(isinstance(asset, dict) and absolute(asset.get("path")), "input path must be absolute")
        path = asset["path"]
        require(path not in by_path, "duplicate input identity: " + path)
        by_path[path] = asset
        sha = asset.get("sha256")
        mode = asset.get("identity_mode", "sha256")
        if sha is not None:
            require(isinstance(sha, str) and re.fullmatch(r"[a-fA-F0-9]{64}", sha), "invalid SHA256: " + path)
            require(mode in ("sha256", "historical_sha256"), "invalid fixed identity mode")
        else:
            require(mode == "stat_then_sha256" and asset.get("role") in ("base", "index"),
                    "only historical base/index assets may lack a pre-existing SHA256")
            stat = asset.get("expected_stat")
            require(isinstance(stat, list) and len(stat) == 3 and all(type(v) is int and v >= 0 for v in stat),
                    "historical size/mtime/inode required before capturing SHA256")
            provenance = asset.get("provenance")
            require(isinstance(provenance, dict) and provenance, "historical stat provenance required")
    curves = spec.get("curves")
    require(isinstance(curves, list) and curves, "no curves selected")
    seen = set()
    identities = set()
    for curve in curves:
        cid = curve.get("id")
        require(isinstance(cid, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,119}", cid)
                and cid not in seen, "invalid or duplicate curve ID")
        seen.add(cid)
        algorithm = curve.get("algorithm")
        require(algorithm in ALGORITHMS, "unsupported native algorithm")
        require(curve.get("workload") in (("origin", "mixed", "p1", "p4", "p7", "p10") if generic else
                                         ("origin", "p1", "p4", "p7", "p10")), "unknown workload")
        require(curve.get("cache_policy") in (("landing", "equal", "off") if generic else
                                             ("landing", "equal") if algorithm == "allianceann" else ("native",)),
                "algorithm/cache mismatch")
        identity = (algorithm, curve["workload"], curve["cache_policy"])
        require(identity not in identities, "duplicate algorithm/workload/cache curve")
        identities.add(identity)
        efs = curve.get("efs")
        require(isinstance(efs, list) and efs and all(positive(v) for v in efs) and sorted(set(efs)) == efs,
                "EFs must be unique, increasing, positive integers")
        require(curve.get("repeats") == aggregation["repeats"] and positive(curve.get("queries")) and positive(curve.get("k"))
                and curve.get("bits") in ((1, 2, 3, 4) if generic else (1,)), "invalid native query protocol")
        argv = curve.get("argv")
        require(isinstance(argv, list) and argv and all(isinstance(v, str) and v and "\x00" not in v for v in argv),
                "native argv must be an explicit nonempty string list")
        if generic:
            # Stored/programmatic plans also reach this validator without the
            # profile adapter. Reject before fingerprinting or creating a job;
            # schema 2 keeps its separately pinned historical driver protocol.
            retired = sorted({arg.split("=", 1)[0] for arg in argv[1:]} & RETIRED_IN_TREE_SEARCH_OPTIONS)
            require(not retired, "retired in-tree search options: " + ", ".join(retired))
            validate_navigation_arguments(argv, by_path)
        require(argv[0] in by_path and by_path[argv[0]].get("sha256") is not None,
                "native binary requires a pinned SHA256")
        env = curve.get("env")
        require(isinstance(env, dict) and all(isinstance(k, str) and re.fullmatch(r"[A-Z][A-Z0-9_]*", k)
                and isinstance(v, str) and "\x00" not in v for k, v in env.items()), "invalid native environment")
        require("LD_PRELOAD" not in env, "LD_PRELOAD must not change the frozen measurement implementation")
        require(curve.get("scope_prefix", "").startswith("alliance-bench-" if generic else SCOPE_PREFIXES[algorithm]),
                "native phase-gate scope prefix mismatch")
        if generic:
            require(algorithm == "allianceann" and curve.get("native_protocol") == "alliance-formal-v1"
                    and curve.get("metric") in ("l2", "ip") and len(efs) <= 20 and efs[-1] <= 1024,
                    "invalid in-tree native curve")
            require(memory.get("mode") in ("phase", "whole_process"), "invalid native memory lifecycle")
            require(env.get("BENCH_EXPECTED_FP") == curve.get("quantization_fingerprint") and
                    str(curve.get("quantization_fingerprint", "")).isdigit(), "unbound quantization fingerprint")
            require(curve.get("input_formats", {}).get("base") in ("fbin", "fvecs") and
                    curve["input_formats"].get("attributes") in ("raw-i32", "ibin"), "explicit input formats required")
        inputs = curve.get("inputs", {})
        require(set(("base", "query", "ranges", "gt", "attributes")) <= set(inputs), "missing independent replay inputs")
        require(all(isinstance(p, str) and p in by_path for p in inputs.values()), "replay input missing from identity ledger")
        require(isinstance(curve.get("expected"), dict), "native evidence expectations must be explicit")
        for text in argv + list(env.values()):
            # Windows sandbox paths can contain a literal UUID in braces.
            literal = re.sub(r"\{[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\}", "", text)
            literal = literal.replace("{output}", "")
            require("{" not in literal and "}" not in literal,
                    "only the {output} runtime placeholder is permitted")
    locks = spec.get("lock_paths")
    require(isinstance(locks, list) and locks and all(absolute(x) for x in locks) and len(set(locks)) == len(locks),
            "explicit, unique global lock paths required")
    require(not (set(locks) & set(by_path)), "lock path must not overwrite an input")
    return spec


def fingerprint_inputs(spec):
    """Check historical identities, then SHA256 every asset before/after a job.

    Old large-index manifests sometimes contain only stat tuples. Those files
    are checked against that tuple before a new hash is captured; the record
    distinguishes this from checking an already-known historical SHA256.
    """
    validate_spec(spec)
    identities = {}
    stats = {}
    for asset in spec["inputs"]:
        path = Path(asset["path"])
        require(path.is_file(), "required asset is missing: " + str(path))
        before = file_stat(path)
        if asset.get("expected_stat") is not None:
            require(before == asset["expected_stat"], "historical input stat changed: " + str(path))
        stats[str(path)] = before
    for asset in spec["inputs"]:
        path = Path(asset["path"])
        before = stats[str(path)]
        digest = digest_file(path)
        require(file_stat(path) == before, "input changed while hashing: " + str(path))
        if asset.get("sha256") is not None:
            require(digest == asset["sha256"].lower(), "input SHA256 mismatch: " + str(path))
        identities[str(path)] = {"sha256": digest, "stat": before,
                                 "identity_mode": asset.get("identity_mode", "sha256"),
                                 "expected_sha256": asset.get("sha256")}
    return identities
