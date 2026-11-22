"""Verify completed build metadata without enabling optional checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def vecs(path, rows, kind="f"):
    with path.open("wb") as out:
        for row in rows:
            out.write(struct.pack("<i" + kind * len(row), len(row), *row))


def execute(command, log, success=True):
    result = subprocess.run(list(map(str, command)), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True, timeout=180)
    log.write_text(result.stdout)
    assert (result.returncode == 0) == success, (command, result.returncode, result.stdout)
    return result.stdout


def suite(build, verifier, root):
    n, dim = 256, 128
    rows = [[math.sin(i * .31 + j * .73) * .2 + (i % 4) * .1 for j in range(dim)]
            for i in range(n)]
    vecs(root / "base.fvecs", rows)
    vecs(root / "centroids.fvecs", rows[:4])
    vecs(root / "cids.ivecs", [[i % 4, (i + 1) % 4] for i in range(n)], "i")
    (root / "attrs.i32").write_bytes(struct.pack("<%di" % n, *reversed(range(n))))
    common = [build, "--basevec", root / "base.fvecs", "--baseatt", root / "attrs.i32",
              "--m", 8, "--efc", 24, "--o", 2, "--wp", 3,
              "--enum_thresh", 32, "--wow_efc", 24, "--threads", 1,
              "--cluster_builder", "wow", "--cluster_partition", "vamana",
              "--cluster_K", 4, "--cluster_wow_efc", 24,
              "--cluster_centroids", root / "centroids.fvecs",
              "--cluster_cids", root / "cids.ivecs", "--rawvec_export", "off"]
    records = []
    for resident in ("default", "explicit"):
        for upper in ("resident", "cluster"):
            directory = root / (resident + "_" + upper)
            directory.mkdir()
            graph, meta = directory / "graph.bin", directory / "index.meta"
            command = common + ["--graph_file", graph, "--meta_file", meta,
                                "--resident_batch", 16,
                                "--wow_thresh", 256 if upper == "resident" else 64]
            if resident == "explicit":
                command += ["--resident_builder", "left_stream"]
            # Deliberately omit --checkpoint_meta. Completion must still persist.
            output = execute(command, directory / "build.log")
            if upper == "cluster":
                assert "Phase2 (cluster) done" in output
            else:
                assert "[Phase2]" not in output
            verify = [verifier, "--mode", "build", "--graph", graph, "--meta", meta,
                      "--base", root / "base.fvecs", "--format", "fvecs",
                      "--attrs", root / "attrs.i32", "--space", "l2"]
            output = execute(verify, directory / "verification.log")
            record = json.loads(next(line for line in output.splitlines() if line.startswith('{"status"')))
            assert record["status"] == "passed"
            assert record["completed_layer"] == record["max_layer"] == 3
            assert record["vectors"] == n and record["node_layers"] == n * 4
            assert record["left_stream"] and record["left_stream_complete"]
            records.append(dict(case=directory.name, verification=record))
    (root / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    print("BUILD_COMPLETION_PASS default/explicit LeftStream resident/cluster no_checkpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="build_completion_", dir=os.getcwd()) as temporary:
        suite(args.build.resolve(), args.verifier.resolve(), Path(temporary))
