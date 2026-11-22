"""Dataset adapters for the in-tree alliance-formal-v1 executable.

Dataset profiles describe geometry and protocol, runtime bindings identify files.
Historical CaseLaw remains on its unchanged, separately validated adapter.
"""
import hashlib
import json
from pathlib import Path

from .protocol import merge, read_json
from .spec import digest_file, require

PROTOCOL = "alliance-formal-v1"


def build_experiment(config_path, bindings_path, workloads=None, algorithms=None, policies=None):
    profile, bindings = read_json(config_path), read_json(bindings_path)
    require(profile.get("schema_version") == 3 and profile.get("native_protocol") == PROTOCOL,
            "unsupported in-tree benchmark profile")
    require(bindings.get("schema_version") == 1 and bindings.get("profile_sha256") == digest_file(config_path),
            "runtime bindings do not pin this dataset profile")
    require(not algorithms or set(algorithms) == {"allianceann"}, "in-tree native adapter supports AllianceANN")
    curves, ledger = [], {}

    def asset(name, role):
        value = dict(bindings["assets"][name])
        require(set(value) <= {"path", "sha256", "expected_stat"} and value.get("sha256"),
                "every dataset asset requires a pinned SHA256")
        value.update(role=role, identity_mode="sha256")
        previous = ledger.setdefault(value["path"], value)
        require(previous["sha256"] == value["sha256"], "conflicting asset identity")
        return value["path"]

    for binding in bindings["curves"]:
        workload, policy = binding["workload"], binding["cache_policy"]
        if workloads and workload not in workloads or policies and policy not in policies:
            continue
        require(workload in profile["workloads"], "workload is absent from dataset profile")
        p = merge(profile["defaults"], profile["workloads"][workload])
        require(type(p.get("ml_only", False)) is bool, "ml_only must be a boolean")
        require(p["threads"] == 1 and p["metric"] in ("ip", "l2"), "single-thread IP/L2 protocol required")
        require(p["quantization_source"] in ("checkpoint", "meta"), "explicit quantization source required")
        require(p["memory"]["mode"] in ("phase", "whole_process"), "explicit memory lifecycle required")
        paths = {role: asset(name, "binary" if role == "binary" else "base" if role == "base" else "input")
                 for role, name in binding["assets"].items()}
        required = {"binary", "base", "attributes", "query", "ranges", "gt", "meta", "ml_graph", "ml_meta"}
        if not p.get("ml_only", False):
            required.add("graph")
        if p["quantization_source"] == "checkpoint":
            required.update(("checkpoint", "centroids", "cids"))
        else:
            require(not set(paths).intersection(("checkpoint", "centroids", "cids")), "meta mode forbids quantization overrides")
        require(required <= set(paths), "missing native or replay asset binding")
        navigation = p.get("navigation", True)
        require(type(navigation) is bool or isinstance(navigation, dict), "invalid navigation profile")
        if navigation is True:
            navigation = {"ef": 64, "budget": 256, "count": 16}
        elif isinstance(navigation, dict):
            require(set(navigation) == {"ef", "budget", "count"}, "explicit navigation ef/budget/count required")
        if navigation:
            require(all(type(v) is int and v > 0 for v in navigation.values()) and
                    navigation["count"] <= min(navigation["ef"], navigation["budget"]) and
                    navigation["ef"] <= 65536 and navigation["budget"] <= 65536,
                    "navigation requires 0 < count <= min(ef, budget), ef/budget <= 65536")
            require("navigation" in paths, "navigation requires a pinned navigation asset")
            require(p["routing"]["filter_mode"] in ("graph", "auto"), "navigation requires graph/auto routing")
        else:
            require("navigation" not in paths, "navigation asset supplied while navigation is disabled")
        require(policy in ("landing", "equal", "off"), "unsupported static cache policy")
        require((policy == "off") == (p["cache"]["static_bytes"] == 0), "static cache policy/budget mismatch")
        efs = binding["efs"]
        args = {"meta_file": paths["meta"], "ml_graph_file": paths["ml_graph"],
                "ml_meta_file": paths["ml_meta"], "query_vec": paths["query"], "query_rng": paths["ranges"],
                "gt_file": paths["gt"], "space": p["metric"], "k": p["k"], "pipeline": 1, "threads": 1,
                "beam_width": p["beam"], "rerank_beam_width": p["rerank_beam"],
                "rerank_batch_blocks": p["rerank_batch_blocks"], "rerank_warm_blocks": 0,
                "filter_mode": p["routing"]["filter_mode"], "prefilter_threshold": p["routing"]["prefilter_threshold"],
                "top_layer_postfilter": int(p["routing"]["top_layer_postfilter"]),
                "entry_sampling": p["entry"]["sampling"], "entry_samples": p["entry"]["samples"],
                "entry_count": p["entry"]["count"], "rabitq_bits": p["bits"], "packed_rabitq": 1,
                "ml_cache_bytes": p["cache"]["dynamic_bytes"], "ml_cache_policy": "fifo",
                "ml_cache_auto": int(p["cache"]["auto_size"]),
                "rerank_policy": "error_bound", "rerank_candidate_factor": 2,
                "rerank_bound_prescreen": 1, "rerank_bound_rank_factor": 1,
                "rerank_float_refine": 1, "rerank_reuse_estimates": 0,
                "memory_budget_bytes": 0, "query_limit": p["queries"], "efs_list": ",".join(map(str, efs))}
        if navigation:
            args.update(navigation_file=paths["navigation"], navigation_ef=navigation["ef"],
                        navigation_budget=navigation["budget"], navigation_count=navigation["count"],
                        entry_sampling="range", entry_samples=0, entry_count=0)
        if "graph" in paths:
            args["graph_file"] = paths["graph"]
        if p["quantization_source"] == "checkpoint":
            args.update(quant_checkpoint=paths["checkpoint"], rabitq_centroids=paths["centroids"], rabitq_cids=paths["cids"])
        argv = [paths["binary"]]
        for key, value in args.items():
            argv += ["--" + key, str(value)]
        if p["cache"]["clear_per_query"]:
            argv.append("--per_query_cache")
        if p.get("ml_only", False):
            argv.append("--ml_only")
        shape = dict(p["shape"])
        for key, value in bindings.get("index_shape", {}).items():
            require(key not in shape or shape[key] == value, "runtime index geometry differs from profile")
            shape[key] = value
        controls = {"OUTPUT": "{output}", "MODE": "measure", "ORDER": p["batch_order"], "QUANTIZATION_SOURCE": p["quantization_source"],
                    "MEMORY_MODE": p["memory"]["mode"], "SCOPE_PREFIX": "alliance-bench-",
                    "EXPECTED_N": shape["vectors"], "EXPECTED_D": shape["dimensions"], "EXPECTED_WP": shape["wp"],
                    "EXPECTED_M": shape["degree"], "EXPECTED_MAX_LAYER": shape["max_layer"],
                    "EXPECTED_CENTROIDS": shape["centroids"], "EXPECTED_QUERIES": p["queries"], "EXPECTED_BITS": p["bits"],
                    "EXPECTED_FP": binding["quantization_fingerprint"], "LOADING_RSS_BYTES": p["memory"]["loading_limit_bytes"],
                    "SEARCH_RSS_BYTES": p["memory"]["search_rss_limit_bytes"], "STATIC_BUDGET": p["cache"]["static_bytes"],
                    "DYNAMIC_BUDGET": p["cache"]["dynamic_bytes"], "REPEATS": p["aggregation"]["repeats"],
                    "WARMUPS": p["warmup_batches"]}
        env = {"BENCH_" + key: str(value) for key, value in controls.items()}
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        if p.get("malloc_arena_max") is not None:
            env["MALLOC_ARENA_MAX"] = str(p["malloc_arena_max"])
        expected = {"measured": {"protocol": PROTOCOL, "edge_block_bytes": shape["edge_bytes"],
                                 "vector_block_bytes": shape["vector_bytes"],
                                 "dynamic_cache_clear_per_query": p["cache"]["clear_per_query"],
                                 "dynamic_cache_auto_size": p["cache"]["auto_size"],
                                 "top_layer_postfilter": p["routing"]["top_layer_postfilter"]},
                    "maximum_measured": {"static_resident_bytes": p["cache"]["static_bytes"],
                                         "dynamic_cache_payload_bytes": p["cache"]["dynamic_bytes"]},
                    "require_zero_prefilter_queries": p["routing"]["filter_mode"] == "graph"}
        if navigation:
            expected["measured"].update(entry_mode="ram_navigation", entry_samples=0, entry_count=0,
                                        guided_entry_bytes=0, navigation_ef=navigation["ef"],
                                        navigation_budget=navigation["budget"], navigation_count=navigation["count"])
        curves.append(dict(id="allianceann-" + workload + "-" + policy, algorithm="allianceann", workload=workload,
                           cache_policy=policy, metric=p["metric"], native_protocol=PROTOCOL,
                           ml_only=p.get("ml_only", False),
                           input_formats=p["input_formats"], shape=shape, efs=efs, repeats=p["aggregation"]["repeats"],
                           warmup_batches=p["warmup_batches"], queries=p["queries"], k=p["k"], bits=p["bits"], threads=1,
                           argv=argv, env=env, scope_prefix="alliance-bench-", expected=expected,
                           quantization_fingerprint=str(binding["quantization_fingerprint"]),
                           inputs={key: paths[key] for key in ("base", "attributes", "query", "ranges", "gt")}))
    require(curves, "no selected dataset curves")
    for path in (config_path, bindings_path):
        path = str(Path(path).resolve())
        ledger[path] = dict(path=path, sha256=digest_file(path), role="configuration", identity_mode="sha256")
    default = profile["defaults"]
    for workload in profile["workloads"]:
        effective = merge(default, profile["workloads"][workload])
        require(effective["memory"] == default["memory"] and effective["aggregation"] == default["aggregation"],
                "one job must share memory and aggregation protocol")
    spec = dict(schema_version=3, native_protocol=PROTOCOL, dataset=profile["dataset"], adapter="dataset-native-v1",
                curves=curves, memory=default["memory"], aggregation=default["aggregation"],
                lock_paths=bindings["lock_paths"], inputs=list(ledger.values()), binding_errors=[],
                production_inputs_checked=False)
    spec["id"] = profile["dataset"] + "-" + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    return spec


def dispatch(config_path, bindings_path, workloads=None, algorithms=None, policies=None):
    if read_json(config_path).get("schema_version") == 3:
        return build_experiment(config_path, bindings_path, workloads, algorithms, policies)
    from .caselaw import build_experiment as historical
    return historical(config_path, bindings_path, workloads, algorithms, policies)
