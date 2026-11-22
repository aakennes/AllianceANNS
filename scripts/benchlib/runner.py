"""One native invocation per curve, with durable, non-replaying recovery."""
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import traceback
import uuid


class RunError(RuntimeError):
    pass


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _sync_directory(path):
    """Persist directory entries on POSIX; errors must prevent execution.

    Windows supports local contract tests only. Native execution uses Linux
    systemd scopes, where skipping a failed directory fsync is never allowed.
    """
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    # Include existing ancestors: a previous interrupted mkdir may have left
    # them visible without persisting the links that make this job reachable.
    for directory in (path,) + tuple(path.parents):
        _sync_directory(directory)


def _write(path, value, exclusive=False):
    path = Path(path)
    content = _canonical(value) + b"\n"
    if exclusive:
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(path.parent)
        return
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _event(job, phase, **fields):
    record = dict(time=_now(), phase=phase, **fields)
    with (job / "journal.jsonl").open("ab") as stream:
        stream.write(_canonical(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(job)


@contextlib.contextmanager
def _locks(paths):
    held = []
    token = uuid.uuid4().hex
    try:
        for path in sorted({str(Path(item).resolve()) for item in paths}):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if os.name == "nt":
                    import msvcrt
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b"\0")
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                raise RunError("Lock is held: " + str(path))
            held.append(fd)
            # Existing experiments use persistent flock inodes. Write owner
            # metadata only after acquiring that same kernel lock; never unlink.
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, _canonical({"token": token, "pid": os.getpid(),
                                    "host": socket.gethostname(), "created_at": _now()}) + b"\n")
            os.fsync(fd)
        yield
    finally:
        for fd in reversed(held):
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _validate(spec):
    from .spec import validate_spec
    return validate_spec(spec)


def _input_fingerprints(spec):
    from .spec import fingerprint_inputs
    return fingerprint_inputs(spec)


def _source_fingerprints():
    package = Path(__file__).resolve().parent
    files = list(package.glob("*.py"))
    cli = package.parent / "benchmark.py"
    if cli.exists():
        files.append(cli)
    return {str(path.relative_to(package.parent)): _sha(path) for path in sorted(files)}


def _fingerprints(spec):
    return {"inputs": _input_fingerprints(spec), "source": _source_fingerprints()}


def clean_environment(explicit):
    """Discard inherited benchmark knobs; only the request can reintroduce them."""
    prefixes = ("CASELAW", "PIPE", "TCMALLOC", "MALLOC_", "STABILITY_", "LD_", "OMP_", "GOMP_", "KMP_",
                "MKL_", "OPENBLAS_", "VECLIB_", "NUMEXPR_", "ALLIANCE", "RABITQ", "BENCH_", "LAION_")
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(prefixes) and key not in ("PYTHONPATH", "PYTHONHOME")}
    env.update({str(key): str(value) for key, value in explicit.items()})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if os.name != "nt":
        # Noninteractive SSH may have no user-bus environment. Bind explicitly
        # to this uid; an inherited path/address from another session is unsafe.
        runtime = "/run/user/" + str(os.getuid())
        env["XDG_RUNTIME_DIR"] = runtime
        env["DBUS_SESSION_BUS_ADDRESS"] = "unix:path=" + runtime + "/bus"
    return env


def _stop_scope(unit, env, directory):
    """Stop only this launch's unit while the experiment locks remain held."""
    cleanup = {"scope_unit": unit, "started_at": _now(), "actions": []}

    def command(arguments, timeout):
        action = {"argv": ["systemctl", "--user"] + arguments}
        try:
            result = subprocess.run(action["argv"], env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)
            action.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except BaseException as error:
            action.update(error=repr(error))
        cleanup["actions"].append(action)
        _write(directory / "cleanup.json", cleanup)
        return action

    def stopped():
        action = command(["show", "--property=ActiveState", "--property=LoadState", unit], 5)
        properties = dict(line.split("=", 1) for line in action.get("stdout", "").splitlines() if "=" in line)
        # Do not treat an unavailable user bus or a failed command as proof.
        return (properties.get("LoadState") == "not-found" or
                (action.get("exit_code") == 0 and properties.get("ActiveState") == "inactive"))

    command(["stop", unit], 10)
    complete = stopped()
    if not complete:
        command(["kill", "--kill-who=all", "--signal=KILL", unit], 5)
        command(["stop", unit], 10)
        complete = stopped()
    cleanup.update(status="stopped" if complete else "failed", finished_at=_now())
    _write(directory / "cleanup.json", cleanup)
    if not complete:
        raise RunError("Could not confirm this scope stopped; inspect cleanup.json and unit " + unit)
    return cleanup


def execute_curve(spec, curve, curve_dir):
    """Launch the full EF/repeat sequence once; native code owns phase gating."""
    curve_dir = Path(curve_dir).resolve()
    sent = _read(curve_dir / "sent.json")
    prefix = curve.get("scope_prefix")
    if not prefix or not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise RunError("A native phase-gate scope prefix is required")
    unit = prefix + sent["run_token"][:20]
    argv = [str(arg).replace("{output}", str(curve_dir / "raw")) for arg in curve["argv"]]
    explicit_env = {key: value.replace("{output}", str(curve_dir / "raw"))
                    for key, value in curve.get("env", {}).items()}
    config = {"argv": argv, "cwd": str(curve_dir), "scope_unit": unit + ".scope",
              "memory_path": str(curve_dir / "memory.json"),
              "loading_limit_bytes": spec["memory"]["loading_limit_bytes"],
              "memory_mode": spec["memory"].get("mode", "phase"),
              "request_sha256": sent["request_sha256"], "curve_id": curve["id"],
              "run_token": sent["run_token"]}
    _write(curve_dir / "scope_request.json", config, exclusive=True)
    command = ["systemd-run", "--user", "--scope", "--quiet", "--unit=" + unit,
               "-p", "MemoryMax=" + str(spec["memory"]["loading_limit_bytes"]),
               "-p", "MemorySwapMax=0", "taskset", "-c", "1", sys.executable,
               str(Path(__file__).with_name("scope_entry.py")),
               "--config", str(curve_dir / "scope_request.json")]
    _write(curve_dir / "launch.json", {"argv": command, "native_argv": argv,
                                      "explicit_env": explicit_env}, exclusive=True)
    env = clean_environment(explicit_env)
    try:
        with (curve_dir / "stdout.log").open("xb") as stdout, (curve_dir / "stderr.log").open("xb") as stderr:
            result = subprocess.run(command, cwd=str(curve_dir), stdout=stdout, stderr=stderr, env=env)
    except BaseException:
        # Killing systemd-run alone does not stop processes owned by its scope.
        # No experiment lock is released until this bounded cleanup completes.
        _stop_scope(unit + ".scope", env, curve_dir)
        raise
    if result.returncode != 0:
        # A failed wrapper may have left descendants in its scope. Stop them
        # without rerunning any native command, and preserve the real exit code.
        _stop_scope(unit + ".scope", env, curve_dir)
    return {"exit_code": result.returncode, "scope_unit": unit + ".scope"}


def _verify_memory(spec, curve, directory, sent, finished):
    path = directory / "memory.json"
    if not path.is_file():
        raise RunError("Missing real scope evidence; sent curves cannot be relaunched")
    cleanup = directory / "cleanup.json"
    if cleanup.exists() and _read(cleanup).get("status") != "stopped":
        raise RunError("Scope cleanup remains unconfirmed; manual inspection is required")
    memory = _read(path)
    required = {"schema_version": 1, "status": "finished", "exit_code": 0,
                "request_sha256": sent["request_sha256"], "curve_id": curve["id"],
                "run_token": sent["run_token"],
                "initial_memory_max": str(spec["memory"]["loading_limit_bytes"]),
                "initial_memory_swap_max": "0", "final_memory_max":
                    str(spec["memory"]["loading_limit_bytes"]) if spec["memory"].get("mode") == "whole_process" else "max",
                "final_memory_swap_max": "0", "affinity_before": [1], "affinity_after": [1]}
    for key, value in required.items():
        if key not in memory or memory[key] != value:
            raise RunError("Scope evidence mismatch: " + key)
    expected_unit = curve.get("scope_prefix", "") + sent["run_token"][:20] + ".scope"
    if memory.get("scope_unit") != expected_unit:
        raise RunError("Scope launch identity mismatch")
    if not memory.get("cgroup_path") or not memory.get("scope_unit"):
        raise RunError("Missing cgroup identity")
    if Path(memory["cgroup_path"]).name != memory["scope_unit"]:
        raise RunError("Scope/cgroup identity mismatch")
    for phase in ("memory_events_before", "memory_events_after"):
        events = memory.get(phase, {})
        if events.get("oom") != 0 or events.get("oom_kill") != 0:
            raise RunError("Missing/nonzero OOM evidence: " + phase)
    if finished is not None and finished.get("exit_code") != 0:
        raise RunError("Native launcher failed; sent curves cannot be relaunched")
    return memory


def _artifacts(directory):
    files = []
    raw = directory / "raw"
    if raw.exists():
        files.extend(path for path in raw.rglob("*") if path.is_file())
    for name in ("stdout.log", "stderr.log", "memory.json", "cleanup.json", "sent.json", "finished.json",
                 "scope_request.json", "launch.json"):
        path = directory / name
        if path.is_file():
            files.append(path)
    result = {}
    for path in sorted(files):
        if path.is_symlink() or directory.resolve() not in path.resolve().parents:
            raise RunError("Evidence must remain inside its curve directory")
        result[path.relative_to(directory).as_posix()] = _sha(path)
    return result


def _verify_curve(spec, curve, directory, verifier, request_hash):
    sent = _read(directory / "sent.json")
    if sent.get("request_sha256") != request_hash or sent.get("curve_sha256") != _digest(curve):
        raise RunError("Sent marker identity changed")
    finished_path = directory / "finished.json"
    finished = _read(finished_path) if finished_path.exists() else None
    if finished and finished.get("run_token") != sent.get("run_token"):
        raise RunError("Finished marker belongs to another launch")
    state_path = directory / "state.json"
    state = _read(state_path) if state_path.exists() else {}
    proof_path = directory / "proof.json"
    if state.get("status") == "verified" or proof_path.exists():
        if not proof_path.exists() or (state.get("status") == "verified" and not state.get("proof_sha256")):
            raise RunError("Verified state has no matching proof")
        if state.get("proof_sha256") and state["proof_sha256"] != _sha(proof_path):
            raise RunError("Verified state has no matching proof")
        # A crash may occur after the proof replace and before state replace.
        # Its identity/artifacts still must match, and verification runs again.
        previous = _read(proof_path)
        if previous.get("request_sha256") != request_hash or previous.get("curve_sha256") != _digest(curve):
            raise RunError("Verification proof identity changed")
        if previous.get("artifacts") != _artifacts(directory):
            raise RunError("Previously verified measurement artifacts changed")
    _verify_memory(spec, curve, directory, sent, finished)
    if verifier is None:
        from .validation import verify_curve
        verifier = verify_curve
    before_verification = _artifacts(directory)
    verified = verifier(spec, curve, directory)
    if not isinstance(verified, dict) or verified.get("status") != "verified" or not isinstance(verified.get("rows"), list):
        raise RunError("Curve verification failed: " + str(verified))
    if not verified["rows"] or any(not isinstance(row, dict) for row in verified["rows"]):
        raise RunError("Verifier returned no structured measurement rows")
    rows = []
    for row in verified["rows"]:
        if row.get("curve_id", curve["id"]) != curve["id"]:
            raise RunError("Verifier changed the curve identity")
        rows.append(dict(row, curve_id=curve["id"]))
    verified = dict(verified, rows=rows)
    if _artifacts(directory) != before_verification:
        raise RunError("Measurement evidence changed during verification")
    proof = {"schema_version": 1, "request_sha256": request_hash,
             "curve_sha256": _digest(curve), "run_token": sent["run_token"],
             "recovered_without_launcher_receipt": finished is None,
             "artifacts": _artifacts(directory), "verification": verified,
             "verified_at": _now()}
    _write(directory / "proof.json", proof)
    _write(state_path, {"status": "verified", "proof_sha256": _sha(directory / "proof.json")})
    return verified["rows"]


def _finalize(spec, directory, rows):
    from .export import finalize_experiment
    return finalize_experiment(spec, directory, rows)


def run_experiment(spec, job_dir, resume=False, executor=None, verifier=None):
    """Resume only collection/verification; a durable sent marker forbids replay."""
    _validate(spec)
    directory = Path(job_dir).resolve()
    _mkdir_durable(directory)
    uid = str(os.getuid()) if hasattr(os, "getuid") else hashlib.sha256(os.environ.get("USERNAME", "user").encode()).hexdigest()[:12]
    lock_paths = [Path(tempfile.gettempdir()) / ("alliance-benchmark-" + uid + ".lock"), directory / ".job.lock"]
    lock_paths.extend(spec.get("lock_paths", []))
    with _locks(lock_paths):
        request_path = directory / "request.json"
        request_hash = _digest(spec)
        if request_path.exists():
            if not resume:
                raise RunError("Existing request requires --resume; no measurements were restarted")
            if _digest(_read(request_path)) != request_hash:
                raise RunError("Immutable request identity changed")
            identity = _read(directory / "identity.json")
            if identity.get("request_sha256") != request_hash:
                raise RunError("Request hash record changed")
        else:
            if resume:
                raise RunError("Cannot resume without an immutable request")
            if any(directory.iterdir()) and any(p.name != ".job.lock" for p in directory.iterdir()):
                raise RunError("New job directory must be empty")
            _write(request_path, spec, exclusive=True)
            _write(directory / "identity.json", {"request_sha256": request_hash}, exclusive=True)
        _event(directory, "resume" if resume else "start", request_sha256=request_hash)
        try:
            fingerprints = _fingerprints(spec)
            before_path = directory / "fingerprints.before.json"
            if before_path.exists():
                if _read(before_path) != fingerprints:
                    raise RunError("Source/binary/input identity changed since the original request")
            else:
                if any(directory.glob("curves/*/sent.json")):
                    raise RunError("Sent measurements lack original fingerprints")
                _write(before_path, fingerprints, exclusive=True)
            if executor is None and verifier is None:
                from .validation import preflight_curve
                for curve in spec["curves"]:
                    ready = preflight_curve(spec, curve)
                    if not isinstance(ready, dict) or ready.get("status") != "ready":
                        raise RunError("Curve preflight did not establish readiness")
                _event(directory, "preflight_passed", curves=len(spec["curves"]))
            rows = []
            for curve in spec["curves"]:
                curve_id = curve["id"]
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", curve_id):
                    raise RunError("Unsafe curve id")
                curve_dir = directory / "curves" / curve_id
                _mkdir_durable(curve_dir)
                if not (curve_dir / "sent.json").exists():
                    if any(curve_dir.iterdir()):
                        raise RunError("Unsent curve has unexplained artifacts")
                    _mkdir_durable(curve_dir / "raw")
                    sent = {"schema_version": 1, "request_sha256": request_hash,
                            "curve_sha256": _digest(curve), "curve_id": curve_id,
                            "run_token": uuid.uuid4().hex, "sent_at": _now()}
                    _write(curve_dir / "sent.json", sent, exclusive=True)
                    _event(directory, "sent", curve_id=curve_id, run_token=sent["run_token"])
                    # The sent receipt is durable before entering any executor.
                    try:
                        result = (executor or execute_curve)(spec, curve, curve_dir)
                        if isinstance(result, int):
                            result = {"exit_code": result}
                        if not isinstance(result, dict) or type(result.get("exit_code")) is not int:
                            raise RunError("Executor must return an integer exit_code")
                        _write(curve_dir / "finished.json", dict(result, run_token=sent["run_token"],
                                                               finished_at=_now()), exclusive=True)
                    except BaseException:
                        _event(directory, "executor_failed", curve_id=curve_id, error=traceback.format_exc())
                        raise
                try:
                    rows.extend(_verify_curve(spec, curve, curve_dir, verifier, request_hash))
                except BaseException:
                    # Keep a prior verified proof/state intact: tampering cannot
                    # downgrade it into a repairable, unverified measurement.
                    if not (curve_dir / "proof.json").exists():
                        _write(curve_dir / "state.json", {"status": "verification_failed", "time": _now()})
                    _event(directory, "verification_failed", curve_id=curve_id, error=traceback.format_exc())
                    raise
                _event(directory, "verified", curve_id=curve_id)
            after = _fingerprints(spec)
            _write(directory / "fingerprints.after.json", after)
            if after != fingerprints:
                raise RunError("Source/binary/input identity changed during execution")
            exports = _finalize(spec, directory, rows)
            _write(directory / "manifest.json", {"schema_version": 1, "status": "verified",
                   "request_sha256": request_hash, "curves": [c["id"] for c in spec["curves"]],
                   "row_count": len(rows), "exports": exports, "completed_at": _now()})
            _event(directory, "complete", rows=len(rows))
            return _read(directory / "manifest.json")
        except BaseException:
            _event(directory, "failed", error=traceback.format_exc())
            try:
                _write(directory / "fingerprints.after.json", _fingerprints(spec))
            except Exception as error:
                _write(directory / "fingerprints.after.json", {"status": "identity_failed", "error": str(error)})
            # Failure evidence is append-only; a later resume may re-verify a
            # sent curve, but can never execute that curve a second time.
            raise
