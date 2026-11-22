"""Build native CaseLaw execution specifications from frozen protocol metadata.

This adapter reads configuration only. It never imports historical runners,
checks production availability, chooses new EFs, or launches a native process.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import posixpath
import re

from .protocol import load_plans, read_json, require


ALGORITHMS = ("allianceann", "pipewow", "pipeannfilter")
POLICIES = ("landing", "equal", "native")
SCOPE_PREFIXES = {"allianceann": "caselaw-rss4-r12-", "pipewow": "caselaw-pw-rss4-",
                  "pipeannfilter": "caselaw-paf-origin-rss3-"}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _selection(values, allowed, label):
    if values is None:
        return set(allowed)
    require(isinstance(values, (list, tuple)) and values and len(values) == len(set(values)), "invalid " + label + " selection")
    require(set(values) <= set(allowed), "unknown " + label + " selection")
    return set(values)


def _metadata_sources(bindings, directory):
    assets = []
    for source in bindings["evidence_files"]:
        path = (directory / source["file"]).resolve()
        try:
            path.relative_to(directory.resolve())
        except ValueError:
            raise ValueError("runtime evidence must remain in its configuration directory")
        require(path.suffix == ".json", "runtime evidence must be JSON metadata")
        require(_hash(path) == source["sha256"], "runtime evidence SHA256 mismatch: " + str(path))
        assets.append(dict(path=str(path), sha256=source["sha256"], role="configuration", identity_mode="sha256"))
    return assets


def _native(profile, binding, assets):
    algorithm = binding["algorithm"]
    require(isinstance(binding["fingerprint"], str) and re.fullmatch(r"[0-9]+", binding["fingerprint"])
            and 0 < int(binding["fingerprint"]) < (1 << 64), "invalid frozen quantization fingerprint")
    paths = {name: assets[asset_id]["path"] for name, asset_id in binding["assets"].items()}
    grid = ",".join(str(ef) for ef in profile["efs"][algorithm])
    budgets = profile["cache_budgets"][algorithm]
    routing = profile["routing"]
    binary = paths["binary"]
    if algorithm == "allianceann":
        parameters = [
            ("graph_file", paths["graph"]), ("meta_file", paths["meta"]),
            ("ml_graph_file", paths["ml_graph"]), ("ml_meta_file", paths["static_cache"]),
            ("query_vec", paths["query"]), ("query_rng", paths["ranges"]), ("gt_file", paths["gt"]),
            ("space", profile["metric"]), ("k", profile["k"]), ("pipeline", 1), ("threads", profile["threads"]),
            ("beam_width", profile["beam"]), ("rerank_beam_width", profile["beam"]),
            ("ml_cache_bytes", budgets["dynamic_bytes"]), ("ml_cache_auto", 0), ("ml_cache_policy", "fifo"),
            ("filter_mode", routing["filter_mode"]), ("prefilter_threshold", routing["prefilter_threshold"]),
            ("rerank_policy", "error_bound"), ("inline_rerank", "off"),
            ("entry_sampling", "guided"), ("entry_samples", 256), ("entry_count", 4),
            ("rabitq_bits", profile["bits"]), ("rabitq_centroids", paths["centroids"]),
            ("rabitq_cids", paths["cids"]), ("quant_checkpoint", paths["checkpoint"]),
            ("memory_budget_bytes", 0), ("query_limit", profile["queries"]), ("efs_list", grid),
            ("top_layer_postfilter", routing["top_layer_postfilter"])]
        argv = [binary]
        for key, value in parameters:
            argv.extend(["--" + key, str(value)])
    elif algorithm == "pipewow":
        prefix = binding["index_prefix"]
        require(paths["index_pages"] == prefix + ".pages" and paths["index_meta"] == prefix + ".meta", "PipeWoW prefix/asset mismatch")
        argv = [binary, prefix, profile["metric"], paths["checkpoint"], paths["centroids"], paths["cids"],
                paths["query"], paths["ranges"], paths["gt"], grid, str(budgets["dynamic_bytes"])]
    else:
        query_dir = posixpath.dirname(paths["query"])
        require(paths["query"] == query_dir + "/query.fvecs" and paths["ranges"] == query_dir + "/query.ranges.ibin"
                and paths["gt"] == query_dir + "/gt.top10.ivecs", "PipeANNFilter needs a bound query directory with canonical filenames")
        require(paths["index_graph"] == binding["index_prefix"] + "_disk.index", "PipeANNFilter prefix/graph mismatch")
        argv = [binary, binding["index_prefix"], query_dir, paths["converted_attributes"], paths["filter"],
                paths["bfs_nodes"], str(budgets["static_bytes"]), "{output}", grid, str(profile["aggregation"]["repeats"])]
    env = dict(OMP_NUM_THREADS=str(profile["threads"]), OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", LC_ALL="C", PYTHONUNBUFFERED="1", CASELAW_OUTPUT="{output}",
               CASELAW_REPEATS=str(profile["aggregation"]["repeats"]),
               CASELAW_EXPECTED_BITS=str(profile["bits"]), CASELAW_EXPECTED_NQ=str(profile["queries"]),
               CASELAW_SEARCH_RSS_BYTES=str(profile["memory"]["search"]["limit_bytes"]),
               CASELAW_LOADING_RSS_BYTES=str(profile["memory"]["loading"]["limit_bytes"]),
               CASELAW_STATIC_BUDGET=str(profile["cache_budgets"]["allianceann"]["static_bytes"]),
               PIPEWOW_DISABLE_CACHE="0", PIPEWOW_WARMUP_QUERIES="0", STABILITY_REPEATS="1", STABILITY_WARMUP="0")
    if algorithm == "pipeannfilter":
        env.update(TCMALLOC_AGGRESSIVE_DECOMMIT="1", TCMALLOC_RELEASE_RATE="10")
    else:
        env["MALLOC_ARENA_MAX"] = "4"
        env["CASELAW_EXPECTED_FP"] = str(binding["fingerprint"])
    require(all(text.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", text) or
                ("{" not in text.replace("{output}", "") and "}" not in text.replace("{output}", ""))
                for text in argv + list(env.values())), "only {output} may remain in native arguments")
    return paths, argv, env


def build_experiment(config_path, bindings_path, workloads=None, algorithms=None, cache_policies=None):
    """Return a schema-v2 spec without reading any production input or binary.

    Missing identities are explicit binding_errors. Planning can display them;
    strict spec validation must reject them before hashing or executing a job.
    Algorithm/policy filters also filter the input ledger and binding errors.
    """
    config_path, bindings_path = Path(config_path).resolve(), Path(bindings_path).resolve()
    bindings = read_json(bindings_path)
    require(bindings["schema_version"] == 1 and bindings["dataset"] == "caselaw", "unsupported CaseLaw runtime bindings")
    require(bindings["profiles_sha256"] == _hash(config_path), "runtime bindings target a different frozen profile catalog")
    selected_algorithms = _selection(algorithms, ALGORITHMS, "algorithm")
    selected_policies = _selection(cache_policies, POLICIES, "cache policy")
    plans = load_plans(config_path, workloads)
    ledger = {}

    def include(asset):
        asset = deepcopy(asset)
        path = asset["path"]
        require(isinstance(path, str) and (path.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", path)), "asset path must be absolute")
        require("{output}" not in path, "input paths may not contain runtime placeholders")
        if path in ledger:
            require(ledger[path].get("sha256") == asset.get("sha256") and ledger[path].get("expected_stat") == asset.get("expected_stat"), "conflicting input identities: " + path)
        else:
            ledger[path] = asset

    for asset in _metadata_sources(bindings, bindings_path.parent):
        include(asset)
    for path in (config_path, bindings_path):
        include(dict(path=str(path), sha256=_hash(path), role="configuration", identity_mode="sha256"))
    curves, errors = [], []
    for plan in plans:
        profile = plan["protocol"]
        for case in plan["cases"]:
            if case["algorithm"] not in selected_algorithms or case["cache_policy"] not in selected_policies:
                continue
            cid = "-".join((plan["workload"], case["algorithm"], case["cache_policy"]))
            require(cid in bindings["curves"], "missing runtime curve binding: " + cid)
            binding = bindings["curves"][cid]
            require((binding["algorithm"], binding["workload"], binding["cache_policy"]) ==
                    (case["algorithm"], plan["workload"], case["cache_policy"]), "runtime curve identity mismatch")
            paths, argv, env = _native(profile, binding, bindings["assets"])
            for asset_id in set(binding["assets"].values()) | set(binding.get("extra_assets", [])):
                asset = bindings["assets"][asset_id]
                include(asset)
                if asset.get("identity_mode") == "unbound":
                    errors.append(cid + ": missing frozen identity for " + asset["path"])
            errors.extend(cid + ": " + reason for reason in binding.get("binding_errors", []))
            expected = deepcopy(binding["expected"])
            expected["require_zero_prefilter_queries"] = profile["routing"]["require_zero_prefilter_queries"] if case["algorithm"] == "allianceann" else False
            if case["algorithm"] == "allianceann":
                expected.setdefault("maximum_measured", {})["dynamic_cache_payload_bytes"] = profile["cache_budgets"]["allianceann"]["dynamic_bytes"]
            elif case["algorithm"] == "pipewow":
                expected.setdefault("measured", {})["cache_payload_bytes"] = profile["cache_budgets"]["pipewow"]["dynamic_bytes"]
            else:
                expected.setdefault("measured", {})["static_cache_bytes"] = profile["cache_budgets"]["pipeannfilter"]["static_bytes"]
            if case["algorithm"] != "pipeannfilter":
                expected.setdefault("measured", {})["requested_repeats"] = profile["aggregation"]["repeats"]
            curve = dict(id=cid, algorithm=case["algorithm"], workload=plan["workload"], cache_policy=case["cache_policy"],
                         efs=list(case["efs"]), repeats=profile["aggregation"]["repeats"], queries=profile["queries"],
                         k=profile["k"], bits=profile["bits"], threads=profile["threads"], argv=argv, env=env,
                         binary_path=paths["binary"], scope_prefix=SCOPE_PREFIXES[case["algorithm"]],
                         inputs={key: paths[key] for key in ("base", "query", "ranges", "gt", "attributes")},
                         expected=expected, source_command=deepcopy(binding["source_command"]),
                         declared_missing=sorted(bindings["assets"][asset_id]["path"] for asset_id in set(binding["assets"].values()) | set(binding.get("extra_assets", []))
                                                 if bindings["assets"][asset_id].get("availability") == "known_missing"))
            curves.append(curve)
            for source in plan["verified_configuration_sources"].values():
                include(dict(path=source["resolved_file"], sha256=source["sha256"], role="configuration", identity_mode="sha256"))
    require(curves, "no CaseLaw curves match the selected workload/algorithm/cache policies")
    spec = dict(schema_version=2, curves=curves,
                memory={"loading_limit_bytes": plans[0]["protocol"]["memory"]["loading"]["limit_bytes"],
                        "search_rss_limit_bytes": plans[0]["protocol"]["memory"]["search"]["limit_bytes"]},
                aggregation={"policy": "qps-paired-trim", "repeats": 5}, inputs=[ledger[p] for p in sorted(ledger)],
                lock_paths=list(bindings["lock_paths"]), binding_errors=sorted(set(errors)),
                adapter="caselaw-native-v1", production_inputs_checked=False)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    spec["id"] = "caselaw-" + digest[:20]
    return spec
