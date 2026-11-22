"""Independent replay of archived CaseLaw results; no historical runner imports.

Only the documented native formats are accepted. NumPy is loaded on replay, not
on CLI import. Input vectors are mapped read-only; exact L2 visits returned IDs.
Scope/cgroup enforcement evidence belongs to runner.py, not phase RSS counters.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def field(mapping, key):
    require(key in mapping, "missing required field: " + key)
    return mapping[key]


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, "invalid integer: " + name)
    return value


def number(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0), "invalid number: " + name)
    return value


def count(mapping, key):
    return integer(field(mapping, key), key)


def csv_count(row, key):
    value = field(row, key)
    require(isinstance(value, str) and value.isdigit(), "invalid CSV integer: " + key)
    return int(value)


def equal(actual, expected, name):
    if type(expected) is bool:
        require(actual is expected, "mismatch: " + name)
    elif type(expected) is int:
        require(type(actual) is int and actual == expected, "mismatch: " + name)
    elif type(expected) is float:
        close(actual, expected, name)
    else:
        require(type(actual) is type(expected) and actual == expected, "mismatch: " + name)


def close(actual, expected, name):
    number(actual, name)
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0), "mismatch: " + name)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), "JSON must be an object: " + str(path))
    return value


def _fingerprint(curve, curve_dir):
    explicit = curve.get("quantization_fingerprint")
    environment = curve.get("env", {}).get("BENCH_EXPECTED_FP" if curve.get("native_protocol") == "alliance-formal-v1"
                                             else "CASELAW_EXPECTED_FP")
    if curve["algorithm"] in ("allianceann", "pipewow"):
        require(explicit is not None or environment is not None, "missing expected quantization fingerprint")
    if explicit is None and environment is None:
        return None
    require(curve["algorithm"] in ("allianceann", "pipewow"), "unsupported fingerprint evidence")
    expected = explicit if explicit is not None else environment
    require(isinstance(expected, str) and expected.isdigit() and int(expected) < (1 << 64),
            "invalid quantization fingerprint")
    if explicit is not None and environment is not None:
        equal(environment, explicit, "fingerprint environment")
    log = (Path(curve_dir) / "stdout.log").read_text(encoding="utf-8")
    markers = re.findall(r"^SEARCH_VERIFIED fingerprint=([0-9]+)\s*$", log, flags=re.MULTILINE)
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    final = re.fullmatch(r"SEARCH_VERIFIED fingerprint=([0-9]+)", lines[-1]) if lines else None
    require(final and markers and int(final.group(1)) == int(expected) and
            all(int(value) == int(expected) for value in markers),
            "missing/mismatched native quantization fingerprint")
    return expected


def _inputs(curve, mappings):
    import numpy as np

    def mapped(path, **kwargs):
        array = np.memmap(str(path), mode="r", **kwargs)
        mappings.append(array)
        return array

    paths = {key: Path(field(curve["inputs"], key))
             for key in ("base", "query", "ranges", "gt", "attributes")}
    formats = curve.get("input_formats", {})
    with paths["base"].open("rb") as stream:
        header = stream.read(8)
    require(len(header) == 8, "short base header")
    if formats.get("base", "fbin") == "fvecs":
        dim = struct.unpack("<I", header[:4])[0]
        require(dim > 0 and paths["base"].stat().st_size % ((dim+1)*4) == 0, "invalid base fvecs length")
        n = paths["base"].stat().st_size // ((dim+1)*4)
        base_raw = mapped(paths["base"], dtype="<u4", shape=(n, dim+1))
        require(bool(np.all(base_raw[:, 0] == dim)), "invalid base fvecs row header")
        base = base_raw[:, 1:].view("<f4")
    else:
        n, dim = struct.unpack("<II", header)
        require(n > 0 and dim > 0 and paths["base"].stat().st_size == 8 + n * dim * 4,
                "invalid base matrix shape")
        base = mapped(paths["base"], dtype="<f4", offset=8, shape=(n, dim))
    if curve.get("native_protocol") == "alliance-formal-v1":
        require(curve["shape"]["vectors"] == n and curve["shape"]["dimensions"] == dim, "profile/input shape mismatch")
    query_bytes = paths["query"].stat().st_size
    require(query_bytes > 0 and query_bytes % ((dim + 1) * 4) == 0, "invalid fvecs length")
    total_queries = query_bytes // ((dim + 1) * 4)
    nq, k = curve["queries"], curve["k"]
    require(nq <= total_queries, "query count exceeds input")
    queries_raw = mapped(paths["query"], dtype="<u4", shape=(total_queries, dim + 1))
    require(bool(np.all(queries_raw[:, 0] == dim)), "invalid fvecs row header")
    queries = queries_raw[:, 1:].view("<f4")[:nq]
    require(bool(np.isfinite(queries).all()), "nonfinite query vector")
    require(paths["ranges"].stat().st_size == total_queries * 8, "invalid ranges length")
    ranges = mapped(paths["ranges"], dtype="<i4", shape=(total_queries, 2))[:nq]
    require(bool(np.all(ranges[:, 0] <= ranges[:, 1])), "reversed predicate range")
    attr_offset = 8 if formats.get("attributes") == "ibin" else 0
    if attr_offset:
        with paths["attributes"].open("rb") as stream:
            require(stream.read(8) == struct.pack("<II", n, 1), "invalid attributes matrix header")
    require(paths["attributes"].stat().st_size == attr_offset + n * 4, "invalid attributes length")
    attributes = mapped(paths["attributes"], dtype="<i4", offset=attr_offset, shape=(n,))
    require(paths["gt"].stat().st_size == total_queries * (k + 1) * 4, "invalid GT length")
    gt_raw = mapped(paths["gt"], dtype="<u4", shape=(total_queries, k + 1))
    require(bool(np.all(gt_raw[:, 0] == k)), "invalid GT row header")
    gt = gt_raw[:nq, 1:]
    require(bool(np.all(gt < n)), "GT ID out of range")
    require(all(len(set(map(int, row))) == k for row in gt), "duplicate GT IDs")
    require(bool(np.all((attributes[gt] >= ranges[:, :1]) & (attributes[gt] <= ranges[:, 1:]))),
            "GT violates predicate")
    return np, base, queries, ranges, gt, attributes


def _results(path, curve, ef, np):
    nq, k = curve["queries"], curve["k"]
    if curve["algorithm"] in ("allianceann", "pipewow"):
        with path.open("rb") as stream:
            header = stream.read(32)
        magic = 0x414c4c49414e4331 if curve.get("native_protocol") == "alliance-formal-v1" else 0x434153454c415731
        require(len(header) == 32 and struct.unpack("<4Q", header) == (magic, nq, k, ef), "invalid fixed-result header")
        require(path.stat().st_size == 32 + nq * k * 12, "short/trailing fixed results")
        with path.open("rb") as stream:
            stream.seek(32)
            raw = np.fromfile(stream, dtype=np.dtype([("id", "<u8"), ("distance", "<f4")]))
        return list(raw.reshape(nq, k))
    rows = []
    dtype = np.dtype([("id", "<u4"), ("distance", "<f4")])
    with path.open("rb") as stream:
        for _ in range(nq):
            header = stream.read(4)
            require(len(header) == 4, "short counted-result header")
            returned = struct.unpack("<I", header)[0]
            require(returned <= k, "too many returned results")
            payload = stream.read(returned * 8)
            require(len(payload) == returned * 8, "short counted-result payload")
            rows.append(np.frombuffer(payload, dtype=dtype))
        require(not stream.read(1), "trailing counted results")
    return rows


def _result_checks(results, observations, curve, inputs):
    np, base, queries, ranges, gt, attributes = inputs
    hits, shorts, total, max_error = 0, 0, 0, 0.0
    for qi, (raw, observation) in enumerate(zip(results, observations)):
        require(csv_count(observation, "query_id") == qi, "nonsequential query ID")
        ids, distances = raw["id"], raw["distance"]
        returned = len(ids)
        if curve["algorithm"] == "pipeannfilter":
            require(csv_count(observation, "returned") == returned, "returned count mismatch")
        require(bool(np.all(ids < len(base))), "result ID out of range")
        require(len(set(map(int, ids))) == returned, "duplicate result IDs")
        require(bool(np.isfinite(distances).all()), "nonfinite result distance")
        require(bool(np.all((attributes[ids] >= ranges[qi, 0]) & (attributes[ids] <= ranges[qi, 1]))),
                "result violates predicate")
        if returned:
            vectors, query = np.asarray(base[ids], dtype=np.float64), queries[qi].astype(np.float64)
            if curve.get("metric", "l2") == "ip":
                exact = 1.0 - np.einsum("ij,j->i", vectors, query)
            else:
                delta = vectors - query
                exact = np.einsum("ij,ij->i", delta, delta)
            require(bool(np.isfinite(exact).all()), "nonfinite exact distance")
            error = np.abs(exact - distances) / np.maximum(1.0, np.abs(exact))
            require(bool(np.all(error <= 1e-4)), "exact {} mismatch".format(curve.get("metric", "L2")))
            max_error = max(max_error, float(error.max()))
        matched = len(set(map(int, ids)) & set(map(int, gt[qi])))
        require(csv_count(observation, "matches") == matched, "query matches mismatch")
        hits += matched
        shorts += returned < curve["k"]
        total += returned
    return hits, shorts, total, max_error


def _sum_columns(observations, measured, columns):
    for column, key in columns:
        require(sum(csv_count(row, column) for row in observations) == count(measured, key),
                "CSV/batch total mismatch: " + key)


def _native_checks(curve, measured, observations, shorts, input_count):
    algorithm = curve["algorithm"]
    common = [("requests", "io_requests"), ("bytes", "io_bytes"), ("matches", "matches")]
    for row in observations:
        requests, requested_bytes = csv_count(row, "requests"), csv_count(row, "bytes")
        require(requested_bytes % 4096 == 0 and requested_bytes >= requests * 4096 and
                (requests > 0 or requested_bytes == 0), "invalid per-query IO geometry")
    if algorithm in ("allianceann", "pipewow"):
        for key in ("invalid", "duplicates", "predicate", "short", "observer_errors", "unknown_fd",
                    "bad_layout", "pending", "cache_entries_before_timing"):
            require(count(measured, key) == 0, "nonzero native correctness counter: " + key)
        common.append(("completions", "io_completions"))
        require(count(measured, "io_requests") == count(measured, "io_completions"), "unfinished IO")
        require(shorts == 0, "fixed-result archive has short results")
    if algorithm == "allianceann":
        generic = curve.get("native_protocol") == "alliance-formal-v1"
        geometry = curve["shape"] if generic else {"edge_bytes": 4096, "vector_bytes": 8192}
        edge_bytes, vector_bytes = geometry["edge_bytes"], geometry["vector_bytes"]
        equal(field(measured, "edge_block_bytes"), edge_bytes, "edge block size")
        equal(field(measured, "vector_block_bytes"), vector_bytes, "vector block size")
        lifecycle = curve["expected"]["measured"] if generic else {"dynamic_cache_clear_per_query": False, "dynamic_cache_auto_size": False}
        for key, expected in (("dynamic_cache_clear_per_query", lifecycle["dynamic_cache_clear_per_query"]), ("dynamic_cache_clear_per_ef", True),
                              ("dynamic_cache_auto_size", lifecycle["dynamic_cache_auto_size"])):
            equal(field(measured, key), expected, key)
        for row in observations:
            requests, graph, vector = csv_count(row, "requests"), csv_count(row, "graph_io"), csv_count(row, "rerank_io")
            require(requests == csv_count(row, "completions") == graph + vector, "mixed IO request mismatch")
            require(csv_count(row, "bytes") == graph * edge_bytes + vector * vector_bytes, "mixed IO byte mismatch")
        common += [("graph_io", "graph_io"), ("rerank_io", "rerank_io")]
        require(count(measured, "observer_graph_requests") == count(measured, "graph_io") and
                count(measured, "observer_vector_requests") == count(measured, "rerank_io"), "observer/native IO mismatch")
        if generic and curve["expected"]["measured"].get("entry_mode") == "ram_navigation":
            require(count(measured, "navigation_bytes") > 0, "RAM navigation was not loaded")
            calls = count(measured, "navigation_queries")
            require(calls == count(measured, "graph_queries"), "navigation/graph query count mismatch")
            require(count(measured, "navigation_scores") <= calls * count(measured, "navigation_budget"),
                    "navigation exceeded score budget")
            require(count(measured, "navigation_entries") <= calls * count(measured, "navigation_count"),
                    "navigation exceeded entry count")
            for key in ("navigation_expansions", "navigation_edges", "navigation_ns"):
                count(measured, key)
        require(count(measured, "prefilter_queries") + count(measured, "graph_queries") == curve["queries"],
                "route coverage mismatch")
    elif algorithm == "pipewow":
        equal(field(measured, "page_bytes"), 8192, "page size")
        for key, expected in (("cache_clear_per_query", False), ("cache_clear_per_ef", True), ("static_cache_bytes", 0)):
            equal(field(measured, key), expected, key)
        for row in observations:
            requests = csv_count(row, "requests")
            require(requests == csv_count(row, "completions"), "PipeWoW unfinished IO")
            require(csv_count(row, "bytes") == requests * 8192, "PipeWoW IO byte mismatch")
            require(csv_count(row, "cache_lookups") == csv_count(row, "cache_hits") + requests, "cache lookup mismatch")
        common += [("cache_hits", "dynamic_hits"), ("cache_lookups", "dynamic_lookups")]
        require(count(measured, "native_io") == count(measured, "io_requests"), "PipeWoW native IO mismatch")
    else:
        equal(field(measured, "observer_verified"), True, "observer_verified")
        require(count(measured, "observer_errors") == count(measured, "result_errors") == 0, "PAF native errors")
        require(count(measured, "short_queries") == shorts, "short query count mismatch")
        common += [("pre", "pre_filter"), ("in", "in_filter"), ("post", "post_filter"), ("cache_hits", "cache_hits")]
        require(sum(count(measured, key) for key in ("pre_filter", "in_filter", "post_filter")) == curve["queries"],
                "PAF route coverage mismatch")
        graph, attr = count(measured, "graph_reads"), count(measured, "attribute_reads")
        graph_bytes, attr_bytes = count(measured, "graph_bytes"), count(measured, "attribute_bytes")
        requested, completed, eof = count(measured, "io_bytes"), count(measured, "completed_bytes"), count(measured, "attribute_eof_reads")
        require(graph + attr == count(measured, "io_requests") and graph_bytes == graph * 8192 and
                graph_bytes + attr_bytes == requested, "PAF IO classification mismatch")
        require(attr_bytes % 4096 == 0 and attr_bytes >= attr * 4096 and
                (attr > 0 or attr_bytes == 0), "PAF attribute request geometry")
        require(eof <= attr and graph_bytes + attr <= completed <= requested, "PAF invalid completion bytes")
        deficit = requested - completed
        if eof == 0:
            require(deficit == 0, "PAF unexplained short IO")
        else:
            # Native attribute file is <u32 N,1> plus N u32 values; EOF reads
            # begin on a 4 KiB boundary and may request multiple sectors.
            tail = (8 + input_count * 4) % 4096
            minimum_short = 4096 - tail if tail else 4096
            require(deficit >= eof * minimum_short and (deficit + eof * tail) % 4096 == 0,
                    "PAF EOF byte deficit mismatch")
        if "io_requests_per_query" in measured:
            close(measured["io_requests_per_query"], count(measured, "io_requests") / curve["queries"], "IO requests/query")
        if "io_4k_equivalent_per_query" in measured:
            close(measured["io_4k_equivalent_per_query"], requested / 4096 / curve["queries"], "4K equivalent/query")
    _sum_columns(observations, measured, common)


def _memory(spec, measured, algorithm):
    limits = field(spec, "memory")
    load_limit = integer(field(limits, "loading_limit_bytes"), "loading limit", 1)
    search_limit = integer(field(limits, "search_rss_limit_bytes"), "search RSS limit", 1)
    load = integer(field(measured, "loading_peak_rss_bytes"), "loading peak RSS", 1)
    key = "search_peak_rss_bytes" if algorithm == "pipeannfilter" else "search_phase_peak_rss_bytes"
    search = integer(field(measured, key), "search peak RSS", 1)
    require(load <= load_limit and search <= search_limit, "phase memory budget exceeded")
    equal(field(measured, "rss_budget_passed"), True, "rss_budget_passed")
    equal(field(measured, "search_rss_limit_bytes"), search_limit, "search RSS limit")
    row = {"loading_peak_rss_bytes": load, "search_peak_rss_bytes": search}
    for native, normalized in (("loading_cgroup_peak_bytes", "loading_cgroup_peak_bytes"),
                               ("search_phase_cgroup_peak_bytes", "search_cgroup_peak_bytes"),
                               ("search_cgroup_peak_bytes", "search_cgroup_peak_bytes"),
                               ("peak_rss_bytes", "peak_rss_bytes")):
        if native in measured:
            row[normalized] = integer(measured[native], native, 1)
    return row


def _verify_curve(spec, curve, curve_dir, mappings):
    """Replay every repeat/EF, returning numeric rows only after all checks pass.

    expected accepts measured (exact native expectations), maximum_measured and
    require_zero_prefilter_queries. Unsupported algorithms/formats fail closed.
    No files are written and no native program or historical runner is imported.
    """
    algorithm = field(curve, "algorithm")
    require(algorithm in ("allianceann", "pipewow", "pipeannfilter"), "unsupported algorithm")
    for key in ("queries", "k", "bits", "repeats"):
        integer(field(curve, key), key, 1)
    require(curve["bits"] in (1, 2, 3, 4, 8), "unsupported RaBitQ bits")
    if algorithm == "pipeannfilter":
        require(curve["bits"] == 1, "PAF frozen native protocol requires one bit")
    efs = field(curve, "efs")
    require(isinstance(efs, list) and efs and len(set(efs)) == len(efs), "invalid EF grid")
    for ef in efs:
        integer(ef, "EF", 1)
    aggregation = field(spec, "aggregation")
    equal(curve["repeats"], field(aggregation, "repeats"), "aggregation repeat count")
    require(field(aggregation, "policy") in ("qps-paired-trim", "metric-wise-trim", "median", "mean", "trimmed_mean"),
            "unsupported aggregation policy")
    if aggregation["policy"] == "qps-paired-trim":
        require(curve["repeats"] == 5, "paired trim requires five repetitions")
    elif aggregation["policy"] in ("metric-wise-trim", "trimmed_mean"):
        require(curve["repeats"] >= 3, "metric-wise trim requires at least three repetitions")
    expected = field(curve, "expected")
    require(isinstance(expected, dict) and set(expected) <= {"measured", "maximum_measured", "require_zero_prefilter_queries"},
            "unsupported validation expectation")
    exact = expected.get("measured", {})
    require(isinstance(exact, dict), "expected.measured must be an object")
    maximum = expected.get("maximum_measured", {})
    require(isinstance(maximum, dict), "expected.maximum_measured must be an object")
    for key, limit in maximum.items():
        number(limit, "maximum_measured." + key)
    zero_pre = expected.get("require_zero_prefilter_queries", False)
    require(type(zero_pre) is bool, "invalid prefilter expectation")
    require(not zero_pre or algorithm != "pipewow", "PipeWoW has no archived prefilter evidence")
    fingerprint = _fingerprint(curve, curve_dir)
    raw_directory = Path(curve_dir) / "raw"
    expected_paths = {"r{}/ef{}/measured.json".format(repeat, ef)
                      for repeat in range(1, curve["repeats"] + 1) for ef in efs}
    measured_paths = list(raw_directory.rglob("measured.json"))
    actual_paths = {path.relative_to(raw_directory).as_posix() for path in measured_paths}
    require(actual_paths == expected_paths and all(path.is_file() and not path.is_symlink()
                                                   for path in measured_paths),
            "measured batch coverage differs from planned repeat/EF grid")
    inputs = _inputs(curve, mappings)
    rows, proofs = [], []
    for repeat in range(1, curve["repeats"] + 1):
        for ef in efs:
            directory = raw_directory / ("r" + str(repeat)) / ("ef" + str(ef))
            measured = _read_json(directory / "measured.json")
            # AA/PW repeated drivers persist both fields. The repeated PAF
            # driver persists repeat, while its historical single-run driver
            # has neither; only that single-run format may omit repeat.
            if algorithm != "pipeannfilter" or curve["repeats"] > 1 or "repeat" in measured:
                equal(field(measured, "repeat"), repeat, "repeat")
            if algorithm != "pipeannfilter" or "requested_repeats" in measured:
                equal(field(measured, "requested_repeats"), curve["repeats"], "requested_repeats")
            # Historical drivers attest in stdout, not measured.json. If a
            # newer driver also persists the fingerprint, both must agree.
            if fingerprint is not None and "quantization_fingerprint" in measured:
                stored = measured["quantization_fingerprint"]
                require(type(stored) in (str, int) and str(stored).isdigit() and
                        int(stored) == int(fingerprint), "measured quantization fingerprint mismatch")
            for key, value in (("queries", curve["queries"]), ("ef", ef)):
                equal(field(measured, key), value, key)
            for key in ("bits", "k"):
                if algorithm != "pipeannfilter" or key in measured:
                    equal(field(measured, key), curve[key], key)
            for key, value in exact.items():
                equal(field(measured, key), value, "expected.measured." + key)
            for key, limit in maximum.items():
                require(number(field(measured, key), key) <= limit, "maximum_measured exceeded: " + key)
            if zero_pre:
                require(count(measured, "pre_filter" if algorithm == "pipeannfilter" else "prefilter_queries") == 0,
                        "unexpected prefilter query")
            with (directory / "queries.csv").open(newline="", encoding="utf-8") as stream:
                observations = list(csv.DictReader(stream))
            require(len(observations) == curve["queries"], "query metrics row count mismatch")
            results = _results(directory / "results.bin", curve, ef, inputs[0])
            hits, shorts, total, max_error = _result_checks(results, observations, curve, inputs)
            require(hits == count(measured, "matches"), "batch matches mismatch")
            _native_checks(curve, measured, observations, shorts, len(inputs[1]))
            wall = number(field(measured, "batch_wall_seconds"), "batch wall seconds", positive=True)
            recall, qps = hits / (curve["queries"] * curve["k"]), curve["queries"] / wall
            latency_key = "latency_ms" if algorithm == "pipeannfilter" else "latency_seconds"
            latencies = [number(float(field(row, latency_key)), latency_key) for row in observations]
            p95 = sorted(latencies)[int(math.ceil(0.95 * curve["queries"])) - 1]
            if latency_key == "latency_seconds":
                p95 *= 1000
            close(field(measured, "recall"), recall, "recall")
            close(field(measured, "qps"), qps, "QPS")
            close(field(measured, "p95_ms"), p95, "P95")
            io_requests, io_bytes = count(measured, "io_requests"), count(measured, "io_bytes")
            row = dict(validated=True, algorithm=algorithm, workload=field(curve, "workload"),
                       cache_policy=field(curve, "cache_policy"), ef=ef, repeat=repeat,
                       queries=curve["queries"], bits=curve["bits"], k=curve["k"], recall=recall, qps=qps,
                       p95_ms=p95, io_requests_per_query=io_requests / curve["queries"],
                       io_4k_equivalent_per_query=io_bytes / 4096 / curve["queries"],
                       io_8k_equivalent_per_query=io_bytes / 8192 / curve["queries"],
                       io_bytes_per_query=io_bytes / curve["queries"], batch_wall_seconds=wall,
                       short_queries=shorts, returned_results_replayed=total)
            distance_name = "ip" if curve.get("metric") == "ip" else "l2"
            row["max_relative_" + distance_name + "_error"] = max_error
            row.update(_memory(spec, measured, algorithm))
            rows.append(row)
            proofs.append(dict(repeat=repeat, ef=ef, returned_results_replayed=total,
                               exact_distance_metric=distance_name, exact_distance_relative_tolerance=1e-4,
                               max_relative_distance_error=max_error,
                               archive_sha256={name: sha256(directory / name)
                                               for name in ("measured.json", "queries.csv", "results.bin")}))
    proof = {"status": "verified", "curve_id": field(curve, "id"), "rows": rows, "proofs": proofs}
    if fingerprint is not None:
        proof["verified_quantization_fingerprint"] = fingerprint
    return proof


def preflight_curve(spec, curve):
    """Check replay dependencies and input shapes before any native query runs.

    Reads only the base header plus query/GT/predicate inputs; no base-vector
    payload is traversed. Importing this module itself still needs no NumPy.
    """
    require(field(curve, "algorithm") in ("allianceann", "pipewow", "pipeannfilter"), "unsupported algorithm")
    for key in ("queries", "k"):
        integer(field(curve, key), key, 1)
    for key in ("loading_limit_bytes", "search_rss_limit_bytes"):
        integer(field(field(spec, "memory"), key), key, 1)
    mappings = []
    try:
        inputs = _inputs(curve, mappings)
        return {"status": "ready", "queries": curve["queries"], "k": curve["k"],
                "dataset_vectors": len(inputs[1]), "dimension": inputs[1].shape[1],
                "input_formats_verified": True}
    finally:
        for array in reversed(mappings):
            array._mmap.close()


def verify_curve(spec, curve, curve_dir):
    """Verify archived raw/r{repeat}/ef{ef} batches without executing native code.

    All NumPy mappings close on success or failure, including shape-check errors.
    """
    mappings = []
    try:
        return _verify_curve(spec, curve, curve_dir, mappings)
    finally:
        for array in reversed(mappings):
            array._mmap.close()
