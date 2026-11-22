#!/usr/bin/env python3
"""Accept a finished disk build; optionally close resume and remove buildvecs.

Default: verify and retain every data artifact. Deletion requires both --apply
and --close-resume-window. This is deliberately separate from build/shuffle.
Only the exact <graph>.buildvecs regular file can be removed; there are no globs,
recursive operations, rawvec cleanup, or implicit reconstruction/retry of a build.
"""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import uuid


def checked_path(value, root=None, exists=True):
    path = Path(os.path.abspath(str(value)))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if current != path or exists:
                raise
            break
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError("symbolic links are not accepted: " + str(current))
    if root is not None and os.path.commonpath([str(root), str(path)]) != str(root):
        raise RuntimeError("artifact path is outside --artifact-root: " + str(path))
    return path


def identity(path, digest=False):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError("artifact must be a regular file with one link: " + str(path))
    result = dict(path=str(path), device=info.st_dev, inode=info.st_ino,
                  bytes=info.st_size, allocated_bytes=info.st_blocks * 512,
                  mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)
    if digest:
        h = hashlib.sha256()
        fd = os.open(str(path), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as source:
            while True:
                chunk = source.read(4 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        if result != identity(path):
            raise RuntimeError("artifact changed while hashing: " + str(path))
        result["sha256"] = h.hexdigest()
    return result


def unchanged(record, check_digest=False):
    # Hash once, then use the same dev/inode/size/mtime/ctime through acceptance.
    # Re-reading a tens-of-GiB temporary file at every gate adds no identity check.
    if check_digest and (type(record.get("sha256")) is not str or
                         len(record["sha256"]) != 64):
        raise RuntimeError("missing frozen artifact digest: " + record["path"])
    current = identity(Path(record["path"]), check_digest)
    expected = record if check_digest else {k: v for k, v in record.items() if k != "sha256"}
    if current != expected:
        raise RuntimeError("artifact identity changed: " + record["path"])


def read_marker(path, root):
    checked_path(path, root)
    frozen = identity(path)
    if frozen["bytes"] > 16 * 1024 * 1024:
        raise RuntimeError("finalization marker exceeds metadata size limit")
    fd = os.open(str(path), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(fd, "r", encoding="utf-8") as source:
        result = json.load(source)
    unchanged(frozen)
    return result


def atomic_json(path, value):
    checked_path(path, exists=False)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        checked_path(path, exists=False)
        os.replace(str(temporary), str(path))
        directory = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextlib.contextmanager
def artifact_lock(path):
    checked_path(path, exists=False)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("artifact lock is not a private regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("build artifacts are locked by another operation")
        yield
    finally:
        os.close(fd)


def checked_verification(value):
    integer_fields = ("vectors", "dimensions", "node_layers", "ordered_edges",
                      "completed_layer", "max_layer")
    if value.get("status") != "passed" or any(type(value.get(k)) is not int for k in integer_fields):
        raise RuntimeError("build verifier did not return complete acceptance evidence")
    if value["vectors"] <= 0 or value["dimensions"] <= 0 or value["max_layer"] < 0:
        raise RuntimeError("invalid completed build geometry")
    if value["max_layer"] == 0:
        raise RuntimeError("max_layer=0 requires independent build-completion proof; "
                           "the default completed_layer=0 cannot prove layer 0 was built; retaining artifacts")
    if value["completed_layer"] != value["max_layer"]:
        raise RuntimeError("build checkpoint is incomplete")
    if value["node_layers"] != value["vectors"] * (value["max_layer"] + 1):
        raise RuntimeError("build verification did not cover every node/layer")
    if type(value.get("left_stream")) is not bool or type(value.get("left_stream_complete")) is not bool:
        raise RuntimeError("missing resident-build completion evidence")
    if value["left_stream"] and not value["left_stream_complete"]:
        raise RuntimeError("left-stream checkpoint is incomplete")
    return value


def verification_command(args, graph, meta):
    return [str(args.verifier), "--mode", "build", "--graph", str(graph),
            "--meta", str(meta), "--base", str(args.base), "--format", args.format,
            "--attrs", str(args.attrs), "--space", args.space]


def verify(args, graph, meta, proof):
    command = verification_command(args, graph, meta)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, check=False)
    evidence = dict(command=command, returncode=result.returncode,
                    stdout=result.stdout, stderr=result.stderr)
    proof["verification"] = evidence
    if result.returncode:
        raise RuntimeError("full build verification failed: " + result.stderr[-2000:])
    records = []
    for line in result.stdout.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and "status" in item:
            records.append(item)
    if len(records) != 1:
        raise RuntimeError("build verifier must emit exactly one acceptance record")
    evidence["result"] = checked_verification(records[0])
    return evidence


def remove_exact_buildvec(target, frozen):
    # Resolve no filename from a manifest: the caller supplied graph and this
    # name was derived once as graph.name + '.buildvecs'. Pin its parent for unlink.
    unchanged(frozen)
    parent_fd = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or (
                info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
                frozen["device"], frozen["inode"], frozen["bytes"],
                frozen["mtime_ns"], frozen["ctime_ns"]):
            raise RuntimeError("buildvec identity changed before unlink")
        os.unlink(target.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def run(args):
    if args.apply and not args.close_resume_window:
        raise RuntimeError("--apply requires --close-resume-window")
    if args.close_resume_window and not args.apply:
        raise RuntimeError("--close-resume-window requires --apply; default verification retains resume")
    root = checked_path(args.artifact_root)
    if not root.is_dir():
        raise RuntimeError("--artifact-root must be a directory")
    graph = checked_path(args.graph, root)
    meta = checked_path(args.meta, root)
    target = checked_path(str(graph) + ".buildvecs", root, exists=False)
    marker = checked_path(str(graph) + ".build.finalized.json", root, exists=False)
    lock = checked_path(str(graph) + ".build.lock", root, exists=False)
    args.base = checked_path(args.base)
    args.attrs = checked_path(args.attrs)
    args.verifier = checked_path(args.verifier)
    request = dict(format=args.format, space=args.space, base=str(args.base),
                   attrs=str(args.attrs), verifier=str(args.verifier), artifact_root=str(root))
    expected_identities = dict(graph=str(graph), meta=str(meta), base=str(args.base),
                               attrs=str(args.attrs), verifier=str(args.verifier))
    evidence_dir = checked_path(str(graph) + ".build.finalization", root, exists=False)
    evidence_dir.mkdir(exist_ok=True)
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
    proof_path = evidence_dir / (stamp + "-" + uuid.uuid4().hex + ".json")
    proof = dict(schema_version=1, started_utc=stamp, status="pending", reason=args.reason,
                 apply=args.apply, close_resume_window=args.close_resume_window,
                 artifact_root=str(root), graph=str(graph), meta=str(meta), target=str(target),
                 marker=str(marker), lock=str(lock), deleted_logical_bytes=0,
                 deleted_allocated_bytes=0)
    try:
        with artifact_lock(lock):
            # A completed marker is idempotent; a crash between marker and unlink
            # may continue cleanup only for the same accepted identities.
            if marker.exists():
                previous = read_marker(marker, root)
                if not args.apply or not previous.get("resume_closed") or previous.get("target") != str(target):
                    raise RuntimeError("build is already finalized; explicit apply is required to inspect/retry cleanup")
                if previous.get("schema_version") != 1 or previous.get("status") not in (
                        "cleanup_pending", "cleanup_failed", "finalized"):
                    raise RuntimeError("unrecognized finalization marker")
                if previous.get("graph") != str(graph) or previous.get("meta") != str(meta):
                    raise RuntimeError("finalization marker is bound to different artifacts")
                if previous.get("request") != request or set(previous.get("accepted_identities", {})) != set(expected_identities):
                    raise RuntimeError("finalization marker has missing or different acceptance inputs")
                if previous.get("verification", {}).get("command") != verification_command(args, graph, meta):
                    raise RuntimeError("finalization marker verifier command differs")
                for key, record in previous["accepted_identities"].items():
                    if record.get("path") != expected_identities[key]:
                        raise RuntimeError("finalization marker identity path differs: " + key)
                    unchanged(record, key in ("meta", "verifier"))
                checked_verification(previous["verification"]["result"])
                frozen = previous["buildvec_identity"]
                if frozen.get("path") != str(target):
                    raise RuntimeError("marker buildvec path mismatch")
                if previous["status"] == "finalized" and target.exists():
                    raise RuntimeError("a new buildvec appeared after finalization")
                if target.exists():
                    unchanged(frozen, True)
                proof.update(verification=previous["verification"], accepted_identities=previous["accepted_identities"],
                             buildvec_identity=frozen, retried_marker=str(marker))
                record = previous
            else:
                if not target.exists():
                    raise RuntimeError("buildvec is missing without a finalization marker")
                accepted = dict(graph=identity(graph), meta=identity(meta, True),
                                base=identity(args.base), attrs=identity(args.attrs),
                                verifier=identity(args.verifier, True))
                frozen = identity(target, True)
                proof.update(accepted_identities=accepted, buildvec_identity=frozen)
                verification = verify(args, graph, meta, proof)
                for identity_record in accepted.values():
                    unchanged(identity_record)
                unchanged(frozen)
                proof["verification"] = verification
                if not args.apply:
                    proof["status"] = "verified_retained"
                    atomic_json(proof_path, proof)
                    return proof_path, proof
                record = dict(schema_version=1, status="cleanup_pending", resume_closed=True,
                              graph=str(graph), meta=str(meta), target=str(target),
                              request=request,
                              accepted_identities=accepted, buildvec_identity=frozen,
                              verification=verification, reason=args.reason, proof=str(proof_path))
                # Durable marker comes first. Even if deletion is interrupted,
                # the build CLI cannot enter a now-closed resume window.
                atomic_json(marker, record)
            if target.exists():
                checked_path(target, root)
                remove_exact_buildvec(target, frozen)
                proof["deleted_logical_bytes"] = frozen["bytes"]
                proof["deleted_allocated_bytes"] = frozen["allocated_bytes"]
            for identity_record in record["accepted_identities"].values():
                unchanged(identity_record)
            record["status"] = "finalized"
            record["cleanup_proof"] = str(proof_path)
            atomic_json(marker, record)
            proof["status"] = "finalized"
            atomic_json(proof_path, proof)
            return proof_path, proof
    except Exception as error:
        proof["status"] = "failed"
        proof["error"] = str(error)
        atomic_json(proof_path, proof)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact-root", "graph", "meta", "base", "attrs", "verifier"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--format", choices=("bin", "fvecs"), required=True)
    parser.add_argument("--space", choices=("l2", "ip"), default="l2")
    parser.add_argument("--reason", default="accepted build; explicit end of resume window")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--close-resume-window", action="store_true")
    args = parser.parse_args()
    try:
        path, proof = run(args)
        print(json.dumps(dict(status=proof["status"], proof=str(path),
                              deleted_logical_bytes=proof["deleted_logical_bytes"],
                              deleted_allocated_bytes=proof["deleted_allocated_bytes"])))
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
