"""Read-only CaseLaw manifest validation; no native drivers are imported or run."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


WORKLOADS = ("origin", "p1", "p4", "p7", "p10")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def at(value, pointer):
    for key in pointer.split("/"):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def verify_sources(directory, sources):
    verified, documents = {}, {}
    for name, source in sources.items():
        path = (directory / source["file"]).resolve()
        try:
            path.relative_to(directory.resolve())
        except ValueError:
            raise ValueError("reference must remain inside the config directory")
        require(path.suffix in (".json", ".hpp"), "reference must be configuration or option declarations")
        require(path.stat().st_size <= 5 * 1024 * 1024, "reference is too large for configuration metadata")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(digest == source["sha256"], "reference SHA256 mismatch: " + str(path))
        verified[name] = dict(source, resolved_file=str(path))
        if path.suffix == ".json":
            documents[name] = json.loads(content.decode("utf-8-sig"))
    return verified, documents


def validate_profile(profile, documents):
    """Validate the frozen protocol separately from availability of production inputs."""
    name = profile["workload"]
    require(name in WORKLOADS and profile["dataset"] == "caselaw", "unknown CaseLaw workload")
    require(profile["bits"] == 1 and profile["threads"] == 1, "CaseLaw profiles require bits=1 and threads=1")
    require(profile["k"] == 10 and profile["queries"] == 6734 and profile["beam"] == 8, "frozen query protocol changed")
    require(profile["metric"] == "l2", "CaseLaw metric must be l2")
    memory = profile["memory"]
    require(memory["loading"] == {"enforcement": "cgroup_memory_max", "limit_bytes": 9 << 30, "swap_max_bytes": 0},
            "loading requires a separate 9 GiB cgroup cap and no swap")
    require(memory["search"] == {"enforcement": "rss_acceptance", "limit_bytes": 3 << 30,
            "cgroup_memory_max": "max", "reset_peak_per_batch": True},
            "search is a 3 GiB RSS acceptance check, not a 3 GiB cgroup cap")
    lifecycle = profile["cache_lifecycle"]
    require(lifecycle == {"dynamic_clear_per_query": False, "dynamic_clear_per_ef": True,
            "dynamic_clear_per_repeat": True, "empty_dynamic_before_timing": True,
            "dynamic_auto_size": False, "static_cache_reused": True}, "frozen cache lifecycle changed")
    aggregate = profile["aggregation"]
    require(aggregate == {"policy": "qps-paired-trim", "repeats": 5, "retained_repeats": 3,
            "rank_by": ["qps", "repeat"], "all_repeats_must_validate": True}, "CaseLaw requires paired five-to-three aggregation")
    require(profile["warmup_queries"] == 0, "formal CaseLaw protocol has no ANN warmup")
    routing = profile["routing"]
    require(routing["filter_mode"] == "auto", "filter mode must remain explicit")
    require(routing["prefilter_threshold"] == (1 if name == "p10" else 8192), "incorrect workload prefilter threshold")
    require(routing["top_layer_postfilter"] == (1 if name == "origin" else 0), "incorrect workload top-layer postfilter")
    require(routing["require_zero_prefilter_queries"] == (name in ("p1", "p10")), "incorrect graph-only verification")
    algorithms = profile["algorithms"]
    require(algorithms and len(set(algorithms)) == len(algorithms), "algorithms must be nonempty and unique")
    require(set(algorithms) == set(profile["efs"]) == set(profile["cache_budgets"]), "algorithm/EF/cache coverage mismatch")
    for algorithm, grid in profile["efs"].items():
        require(len(grid) == 20 and all(type(ef) is int and ef > 0 for ef in grid)
                and sorted(set(grid)) == grid, "each frozen curve requires 20 distinct increasing positive EFs: " + algorithm)
    require(profile["cache_budgets"]["allianceann"]["dynamic_bytes"] == 8 << 20, "AllianceANN dynamic cache must be 8 MiB")
    require(profile["cache_policies"] == (["landing", "equal"] if name == "p10" else ["landing"]), "cache policy mismatch")
    required_bindings = {"bits", "threads", "beam", "queries", "memory/loading/limit_bytes",
                         "memory/search/limit_bytes", "aggregation/repeats", "aggregation/retained_repeats",
                         "cache_budgets/allianceann/static_bytes", "cache_budgets/allianceann/dynamic_bytes"}
    required_bindings.update("efs/" + algorithm for algorithm in algorithms)
    require(required_bindings <= set(profile["request_bindings"]), "missing frozen evidence binding")
    for field, binding in profile["request_bindings"].items():
        expected = at(documents[binding["source"]], binding["pointer"])
        require(at(profile, field) == expected, "profile differs from frozen evidence: " + field)
    if name == "p10":
        require(profile["efs"]["allianceann"] == documents["efs"]["efs"], "P10 request and selected EF metadata disagree")
        require(profile["sources"]["efs"]["sha256"] == documents["request"]["selected_efs_sha256"], "P10 selected EF identity mismatch")
    require(profile["driver_dependencies"], "frozen driver dependencies must be documented")
    for dependency in profile["driver_dependencies"]:
        require(dependency["path"] and dependency["role"], "incomplete historical dependency")
        digest = dependency.get("sha256")
        require(digest is None or re.fullmatch("[0-9a-f]{64}", digest), "invalid dependency digest")
    return profile


def expand_paths(value, roots):
    if isinstance(value, dict):
        return {key: expand_paths(item, roots) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_paths(item, roots) for item in value]
    if isinstance(value, str):
        for key, root in roots.items():
            value = value.replace("{" + key + "}", root.rstrip("/\\"))
    return value


def load_plans(config_path, workloads=None, path_overrides=None):
    config_path = Path(config_path)
    catalog = read_json(config_path)
    require(catalog["schema_version"] == 1 and catalog["mode"] == "manifest-only", "unsupported planner schema or mode")
    require(set(catalog["profiles"]) == set(WORKLOADS), "profiles must cover origin/p1/p4/p7/p10")
    roots = dict(catalog["path_roots"])
    overrides = path_overrides or {}
    require(isinstance(overrides, dict), "path overrides must be a JSON object")
    require(set(overrides) <= set(roots), "unknown path root override")
    roots.update(overrides)
    require(all(isinstance(value, str) and (value.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", value)) for value in roots.values()),
            "path roots must be absolute")
    selected = list(WORKLOADS if workloads is None else workloads)
    require(selected and len(set(selected)) == len(selected) and set(selected) <= set(WORKLOADS), "invalid workload selection")
    plans = []
    for name in selected:
        profile = merge(catalog["common"], catalog["profiles"][name])
        require(profile["workload"] == name, "workload key mismatch")
        sources, documents = verify_sources(config_path.parent, profile["sources"])
        validate_profile(profile, documents)
        profile.pop("sources")
        profile.pop("request_bindings")
        dependencies = expand_paths(profile.pop("driver_dependencies"), roots)
        inputs = expand_paths(profile.pop("input_paths"), roots)
        cases = [{"algorithm": algorithm, "cache_policy": policy, "efs": profile["efs"][algorithm],
                  "repeats_per_ef": 5, "batches": 100} for algorithm in profile["algorithms"]
                 for policy in (profile["cache_policies"] if algorithm == "allianceann" else ["native"])]
        plans.append({"mode": "manifest-only", "workload": name, "configuration_validated": True,
                      "production_inputs_checked": False, "execution_supported": False,
                      "protocol": profile, "cases": cases, "formal_batches": sum(case["batches"] for case in cases),
                      "formal_queries": sum(case["batches"] for case in cases) * profile["queries"],
                      "path_roots": roots, "input_paths": inputs, "historical_driver_dependencies": dependencies,
                      "verified_configuration_sources": sources,
                      "limitations": ["Frozen drivers still depend on historical experiment paths.",
                                      "Path overrides only change this plan; driver relocatability is not verified.",
                                      "No production input, binary, cache, or query has been opened or executed."]})
    return plans
