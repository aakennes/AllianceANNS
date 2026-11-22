"""Capture actual cgroup v2 evidence inside the native process's user scope."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _events(root):
    return {key: int(value) for key, value in
            (line.split() for line in (root / "memory.events").read_text().splitlines())}


def _cgroup():
    entries = [line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines()
               if line.startswith("0::")]
    if len(entries) != 1:
        raise RuntimeError("A unique cgroup v2 membership is required")
    root = (Path("/sys/fs/cgroup") / entries[0].lstrip("/")).resolve()
    if Path("/sys/fs/cgroup").resolve() not in root.parents:
        raise RuntimeError("Invalid cgroup path")
    return root


def _terminate_child(child):
    cleanup = {"pid": child.pid, "started_at": _now(), "actions": []}
    try:
        if child.poll() is None:
            child.terminate()
            cleanup["actions"].append("terminate")
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                cleanup["actions"].append("kill")
                child.wait(timeout=5)
        cleanup.update(status="reaped", return_code=child.returncode, finished_at=_now())
    except BaseException as error:
        cleanup.update(status="failed", error=repr(error), finished_at=_now())
    return cleanup


def run(config):
    record = {"schema_version": 1, "status": "starting", "started_at": _now(),
              "scope_unit": config["scope_unit"], "request_sha256": config["request_sha256"],
              "curve_id": config["curve_id"], "run_token": config["run_token"], "exit_code": None}
    path = config["memory_path"]
    child = None
    try:
        root = _cgroup()
        record.update(cgroup_path=str(root),
                      initial_memory_max=(root / "memory.max").read_text().strip(),
                      initial_memory_swap_max=(root / "memory.swap.max").read_text().strip(),
                      memory_events_before=_events(root), affinity_before=sorted(os.sched_getaffinity(0)))
        if root.name != config["scope_unit"]:
            raise RuntimeError("Unexpected systemd scope")
        if record["initial_memory_max"] != str(config["loading_limit_bytes"]) or record["initial_memory_swap_max"] != "0":
            raise RuntimeError("Loading cap or swap policy was not applied")
        if record["affinity_before"] != [1]:
            raise RuntimeError("Expected CPU affinity [1]")
        if any(record["memory_events_before"].get(key) != 0 for key in ("oom", "oom_kill")):
            raise RuntimeError("Missing/nonzero initial OOM counters")
        record["status"] = "running"
        _write(path, record)
        child = subprocess.Popen(config["argv"], cwd=config["cwd"])
        record["exit_code"] = child.wait()
        record.update(final_memory_max=(root / "memory.max").read_text().strip(),
                      final_memory_swap_max=(root / "memory.swap.max").read_text().strip(),
                      memory_events_after=_events(root), affinity_after=sorted(os.sched_getaffinity(0)),
                      finished_at=_now())
        peak = root / "memory.peak"
        if peak.exists():
            record["overall_peak_bytes"] = int(peak.read_text().strip())
        final_limit = str(config["loading_limit_bytes"]) if config.get("memory_mode") == "whole_process" else "max"
        if record["final_memory_max"] != final_limit or record["final_memory_swap_max"] != "0":
            raise RuntimeError("Native phase gate did not release the loading cap")
        if record["affinity_after"] != [1] or any(record["memory_events_after"].get(key) != 0 for key in ("oom", "oom_kill")):
            raise RuntimeError("CPU affinity or OOM acceptance failed")
        record["status"] = "finished"
        _write(path, record)
        return record["exit_code"] if record["exit_code"] >= 0 else 128 - record["exit_code"]
    except BaseException as error:
        if child is not None:
            record["child_cleanup"] = _terminate_child(child)
        record.update(status="failed", error=str(error), finished_at=_now())
        _write(path, record)
        print("scope evidence failed: " + str(error), file=sys.stderr)
        return 125


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
