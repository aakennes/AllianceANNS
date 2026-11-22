"""Synthetic acceptance for optional rawvec export and explicit finalization."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command, ok=True):
    result = subprocess.run(list(map(str, command)), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True, timeout=180)
    if (result.returncode == 0) != ok:
        raise AssertionError("command outcome differs: %r\n%s\n%s" % (command, result.stdout, result.stderr))
    return result


def vecs(path, rows, kind="f"):
    with path.open("wb") as out:
        for row in rows:
            out.write(struct.pack("<i" + kind * len(row), len(row), *row))


def vector_path_offset(path):
    data = path.read_bytes()
    header = struct.unpack_from("<11Q", data)
    n = header[6]
    offset = 88 + header[10] * 8 + n * (8 + 4 + 4 + 4)
    quantized = struct.unpack_from("<?", data, offset)[0]
    offset += 1
    if quantized:
        padded, code_bytes, centroids = struct.unpack_from("<3Q", data, offset)
        offset += 24 + padded * centroids * 4 + n + n * code_bytes + n * 16
    size = struct.unpack_from("<Q", data, offset)[0]
    return offset, size, data[offset + 8:offset + 8 + size].decode(), header


def run(args, root):
    n, dim = 128, 128
    rows = [[math.sin(i * .173 + j * .031) + i * .013 for j in range(dim)] for i in range(n)]
    # Ground truth and binary inputs use the same rounded FP32 values.
    rows = [list(struct.unpack("<%df" % dim, struct.pack("<%df" % dim, *row))) for row in rows]
    vecs(root / "base", rows)
    attrs = [(i * 29) % n for i in range(n)]
    (root / "attrs").write_bytes(struct.pack("<%di" % n, *attrs))
    vecs(root / "centroids", [[0.0] * dim])
    vecs(root / "cids", [[0] for _ in rows], "i")
    queries = [rows[i] for i in [0, 19, 71, 115]]
    vecs(root / "query", queries)
    # The search CLI consumes headerless int32 (lo, hi) pairs, unlike GT ivecs.
    (root / "ranges").write_bytes(b"".join(struct.pack("<ii", 0, n - 1) for _ in queries))
    gt = [sorted(range(n), key=lambda i: sum((a-b)**2 for a, b in zip(row, rows[i])))[:10] for row in queries]
    vecs(root / "gt", gt, "i")
    common = [args.build, "--basevec", root / "base", "--baseatt", root / "attrs",
              "--centroids", root / "centroids", "--cids", root / "cids",
              "--m", 8, "--efc", 32, "--o", 2, "--wp", 2, "--enum_thresh", 64,
              "--wow_thresh", 256, "--wow_efc", 32, "--threads", 1,
              "--cluster_builder", "wow", "--resident_builder", "left_stream"]
    cases = {}
    for mode in ("default", "off"):
        directory = root / mode
        directory.mkdir()
        graph, meta = directory / "graph", directory / "meta"
        command = common + ["--graph_file", graph, "--meta_file", meta,
                            "--checkpoint_meta", directory / "checkpoint"]
        if mode == "off":
            Path(str(graph) + ".rawvecs").write_bytes(b"existing rawvec must remain unchanged")
            command += ["--rawvec_export", "off"]
        result = execute(command)
        (directory / "build.log").write_text(result.stdout + result.stderr)
        offset, size, raw_path, header = vector_path_offset(meta)
        assert bool(size) == (mode == "default")
        assert header[2] == 2 and header[6] == n
        cases[mode] = (graph, meta, command)
    graph, meta, command = cases["off"]
    default_graph, default_meta, _ = cases["default"]
    assert sha(graph) == sha(default_graph), "raw export changed graph bytes"
    normalized = []
    for path in (meta, default_meta):
        start, size, _, _ = vector_path_offset(path)
        data = path.read_bytes()
        normalized.append(data[:start] + data[start+8+size:])
    assert normalized[0] == normalized[1], "raw export changed non-path metadata"

    target = Path(str(graph) + ".buildvecs")
    marker = Path(str(graph) + ".build.finalized.json")
    raw = Path(str(graph) + ".rawvecs")
    original = {str(path): sha(path) for path in (graph, meta, raw, target)}
    finalize = [sys.executable, args.finalizer, "--artifact-root", graph.parent,
                "--graph", graph, "--meta", meta, "--base", root / "base",
                "--attrs", root / "attrs", "--format", "fvecs", "--space", "l2",
                "--verifier", args.verifier]
    apply = ["--apply", "--close-resume-window"]
    default = execute(finalize)
    assert json.loads(default.stdout)["status"] == "verified_retained"
    assert target.exists() and not marker.exists()
    execute(finalize + ["--apply"], False)

    # Shared lock excludes the builder too, including resumed writes.
    with Path(str(graph) + ".build.lock").open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert "locked" in execute(finalize + apply, False).stderr
        assert "locked" in execute(command + ["--resume"], False).stderr
    alias = root / "alias"
    alias.symlink_to(graph.parent, target_is_directory=True)
    alias_command = list(command)
    alias_command[alias_command.index("--graph_file") + 1] = alias / "graph"
    assert "symbolic" in execute(alias_command + ["--resume"], False).stderr
    alias.unlink()

    outside = list(finalize)
    outside[outside.index("--artifact-root") + 1] = default_graph.parent
    execute(outside + apply, False)
    moved = target.with_name("saved.buildvecs")
    target.rename(moved)
    target.symlink_to(moved)
    execute(finalize + apply, False)
    assert moved.exists()
    target.unlink(); moved.rename(target)
    hardlink = target.with_name("hardlinked.buildvecs")
    os.link(str(target), str(hardlink))
    execute(finalize + apply, False)
    hardlink.unlink()

    # A checkpoint with unfinished layers must never authorize cleanup.
    data = bytearray(meta.read_bytes())
    start, size, _, _ = vector_path_offset(meta)
    struct.pack_into("<Q", data, start + 8 + size, 0)
    incomplete = meta.with_name("incomplete.meta")
    incomplete.write_bytes(data)
    bad = list(finalize)
    bad[bad.index("--meta") + 1] = incomplete
    execute(bad + apply, False)
    # Verification failure retains full diagnostics and data.
    bad_attrs = graph.parent / "wrong.attrs"
    bad_attrs.write_bytes(struct.pack("<%di" % n, *([999] * n)))
    bad = list(finalize)
    bad[bad.index("--attrs") + 1] = bad_attrs
    execute(bad + apply, False)
    failures = [json.loads(path.read_text()) for path in Path(str(graph)+".build.finalization").glob("*.json")]
    assert any(item.get("verification", {}).get("returncode", 0) != 0 for item in failures)
    assert not marker.exists()
    assert all(sha(Path(path)) == digest for path, digest in original.items())

    # Layer-zero progress defaults to zero even before a layer is built. A
    # geometry-only equality is insufficient evidence to close its resume window.
    zero = root / "zero_layer"
    zero.mkdir()
    zero_build = list(common)
    zero_build[zero_build.index("--wp") + 1] = 0
    execute(zero_build + ["--graph_file", zero / "graph", "--meta_file", zero / "meta"])
    zero_finalize = list(finalize)
    for flag, value in (("--artifact-root", zero), ("--graph", zero / "graph"),
                        ("--meta", zero / "meta")):
        zero_finalize[zero_finalize.index(flag) + 1] = value
    refused = execute(zero_finalize + apply, False)
    assert "independent build-completion proof" in refused.stderr
    assert (zero / "graph.buildvecs").exists() and not (zero / "graph.build.finalized.json").exists()

    # Crash window: accepted marker is durable, but no unlink has happened.
    spec = importlib.util.spec_from_file_location("artifact_finalizer", str(args.finalizer))
    finalizer = importlib.util.module_from_spec(spec)
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(finalizer)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    invocation = SimpleNamespace(artifact_root=graph.parent, graph=graph, meta=meta,
                                 base=root / "base", attrs=root / "attrs", format="fvecs",
                                 space="l2", verifier=args.verifier, reason="synthetic interruption",
                                 apply=True, close_resume_window=True)
    original_remove = finalizer.remove_exact_buildvec
    def fail_before_unlink(*unused):
        raise OSError("injected interruption before unlink")
    finalizer.remove_exact_buildvec = fail_before_unlink
    try:
        finalizer.run(invocation)
        raise AssertionError("injected finalization failure was ignored")
    except OSError as error:
        assert str(error) == "injected interruption before unlink"
    finally:
        finalizer.remove_exact_buildvec = original_remove
    assert target.exists() and json.loads(marker.read_text())["status"] == "cleanup_pending"
    assert sha(target) == original[str(target)]
    assert "resume window is closed" in execute(command + ["--resume"], False).stderr

    result = json.loads(execute(finalize + apply).stdout)
    proof = json.loads(Path(result["proof"]).read_text())
    saved_marker = marker.read_bytes()
    assert result["status"] == "finalized" and not target.exists()
    assert proof["buildvec_identity"]["sha256"] == original[str(target)]
    assert result["deleted_logical_bytes"] == proof["buildvec_identity"]["bytes"]
    assert all(sha(Path(path)) == digest for path, digest in original.items() if path != str(target))
    assert "resume window is closed" in execute(command + ["--resume"], False).stderr
    repeat = json.loads(execute(finalize + apply).stdout)
    assert repeat["status"] == "finalized" and repeat["deleted_logical_bytes"] == 0
    corrupt = json.loads(saved_marker)
    corrupt["accepted_identities"] = {}
    marker.write_text(json.dumps(corrupt))
    execute(finalize + apply, False)
    marker.write_bytes(saved_marker)
    changed_request = list(finalize)
    changed_request[changed_request.index("--space") + 1] = "ip"
    execute(changed_request + apply, False)

    # Buildvec/rawvec are not shuffle sources; shuffled exact vectors still work.
    ml = graph.parent / "ml"
    layout = Path(str(ml) + ".layout")
    shuffle_metadata = graph.parent / "shuffle.meta"
    execute([args.shuffle, "--graph_file", graph, "--meta_file", meta,
             "--out_graph_file", ml, "--out_meta_file", shuffle_metadata, "--multi_level",
             "--method", "identity", "--shuffle_o", 2, "--block_size", 4096])
    execute([args.verifier, "--graph", graph, "--meta", meta, "--ml-graph", ml,
             "--layout", layout, "--base", root / "base", "--format", "fvecs",
             "--attrs", root / "attrs", "--space", "l2"])
    search = execute([args.search, "--graph_file", graph, "--meta_file", meta,
                      "--ml_graph_file", ml, "--ml_meta_file", layout,
                      "--query_vec", root / "query", "--query_rng", root / "ranges",
                      "--gt_file", root / "gt", "--k", 10, "--efs", 64, "--threads", 1,
                      "--pipeline", 1, "--filter_mode", "prefilter", "--prefilter_rerank", n,
                      "--ml_cache_auto", 0, "--ml_cache_bytes", 0])
    assert re.search(r"unique_recall=1(?:\.0+)?\s", search.stdout), search.stdout
    assert "short_results=0 queries=4" in search.stdout, search.stdout
    print("BUILD_ARTIFACTS_PASS raw_default/off byte_parity retained/apply lock incomplete symlink hardlink boundary retry shuffle search")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("build", "shuffle", "search", "verifier", "finalizer"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(args, key, value.resolve())
    with tempfile.TemporaryDirectory(prefix="build_artifacts_", dir=os.getcwd()) as directory:
        run(args, Path(directory))
