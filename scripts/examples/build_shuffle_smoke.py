"""Synthetic acceptance at production dimensions, never production query data."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import numpy as np


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def rows(path, values, dtype):
    values = np.asarray(values, dtype=dtype)
    with path.open('wb') as f:
        for row in values:
            f.write(np.int32(len(row)).tobytes())
            f.write(row.tobytes())


def execute(root, stage, args, cpu='1-32', memory=None):
    cmd = list(map(str, args))
    if cpu:
        cmd = ['taskset', '-c', cpu] + cmd
    if memory:
        cmd = ['systemd-run', '--user', '--scope', '--quiet',
               '-p', 'MemoryMax=' + str(memory), '-p', 'MemorySwapMax=0'] + cmd
    start = time.monotonic()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (root / (stage + '.log')).open('w') as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           cwd=str(root), env=env, timeout=1200)
    record = dict(command=cmd, elapsed_seconds=time.monotonic() - start, returncode=p.returncode)
    dump(root / (stage + '.json'), record)
    if p.returncode:
        raise RuntimeError(stage + ' failed: ' + (root / (stage + '.log')).read_text()[-3000:])
    return record


def prepare(root, dim):
    n, nq, nc = 4096, 32, (256 if dim == 384 else 128)
    rng = np.random.RandomState(20261114 + dim)
    base = rng.normal(size=(n, dim)).astype('<f4')
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    attrs = rng.permutation(n).astype('<i4')
    query = base[rng.choice(n, nq, replace=False)].copy()
    query += rng.normal(scale=.005, size=query.shape).astype('<f4')
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    centers = base[rng.choice(n, nc, replace=False)].copy()
    cids = np.argmax(base @ centers.T, axis=1).astype('<i4')[:, None]
    with (root / 'base.fbin').open('wb') as f:
        f.write(np.array([n, dim], dtype='<u4').tobytes()); f.write(base.tobytes())
    (root / 'attrs.i32').write_bytes(attrs.tobytes())
    rows(root / 'centroids.fvecs', centers, '<f4')
    rows(root / 'cids.ivecs', cids, '<i4')
    rows(root / 'query.fvecs', query, '<f4')
    # Mixed widths exercise routing and predicates without claiming dataset selectivity.
    ranges = []
    for i in range(nq):
        width = [n, n // 2, n // 8, n // 32][i % 4]
        lo = int(rng.randint(n - width + 1))
        ranges.append([lo, lo + width - 1])
    ranges = np.array(ranges, dtype='<i4')
    rows(root / 'ranges.ivecs', ranges, '<i4')
    # Native drivers read plain two-int ranges; keep both explicit representations.
    (root / 'ranges.bin').write_bytes(ranges.tobytes())
    gt = []
    for q, (lo, hi) in zip(query, ranges):
        ids = np.where((attrs >= lo) & (attrs <= hi))[0]
        ids = ids[np.argsort(-(base[ids] @ q), kind='stable')[:10]]
        gt.append(ids)
    rows(root / 'gt.ivecs', gt, '<i4')
    inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    dump(root / 'inputs.json', dict(seed=20261114 + dim, n=n, dim=dim, queries=nq,
         centroid_count=nc, synthetic=True, normalized=True, hashes=inputs))


def run(a):
    root = a.output.resolve()
    if a.finalizer:
        a.finalizer = a.finalizer.resolve()
    root.mkdir(parents=True, exist_ok=a.resume)
    bindir = a.bin.resolve()
    for name, dim, wp, gb in [
            ('arxiv2m', 384, 5, 1), ('laion10m', 512, 6, 5)]:
        job = root / name; job.mkdir(exist_ok=a.resume)
        if not (job / 'inputs.json').exists(): prepare(job, dim)
        common = [bindir / 'alliance_build_disk', '--basevec', job / 'base.fbin',
            '--basevec_format', 'bin', '--baseatt', job / 'attrs.i32', '--m', 32,
            '--efc', 256, '--o', 4, '--wp', wp, '--win_anchor', 'top', '--threads', 32,
            '--space', 'ip', '--rabitq_bits', 3, '--centroids', job / 'centroids.fvecs',
            '--cids', job / 'cids.ivecs', '--node_page_bytes', 4096, '--enum_thresh', 2048,
            '--build_mem_gb', gb, '--wow_efc', 256,
            '--resident_builder', 'left_stream', '--resident_batch', 256,
            '--cluster_builder', 'wow', '--cluster_partition', 'vamana',
            '--cluster_K', 0,
            '--cluster_wow_efc', 256, '--cluster_wow_passes', 1,
            '--graph_file', job / 'graph.bin', '--meta_file', job / 'meta.bin',
            '--checkpoint_meta', job / 'checkpoint.meta']
        if a.no_raw: common += ['--rawvec_export', 'off']
        if not (a.resume and (job / 'build.json').exists() and json.loads((job / 'build.json').read_text())['returncode']==0):
            execute(job, 'memory_plan', common + ['--build_mem_report'], memory=gb * 1024**3)
            execute(job, 'build', common, memory=gb * 1024**3)
        # Acceptance occurs before any explicitly requested artifact finalization.
        if a.finalizer:
            before = {p.name: dict(logical=p.stat().st_size, allocated=p.stat().st_blocks * 512)
                      for p in job.glob('graph.bin*') if p.is_file()}
            finalize = [sys.executable, a.finalizer, '--artifact-root', job,
                '--graph', job / 'graph.bin', '--meta', job / 'meta.bin',
                '--base', job / 'base.fbin', '--format', 'bin',
                '--attrs', job / 'attrs.i32', '--space', 'ip',
                '--verifier', bindir / 'alliance_verify_layout']
            if not (job / 'finalize.json').exists():
                execute(job, 'finalize_preview', finalize, cpu='1')
                assert (job / 'graph.bin.buildvecs').is_file()
                execute(job, 'finalize', finalize + ['--apply', '--close-resume-window'], cpu='1')
                assert not (job / 'graph.bin.buildvecs').exists()
                if a.no_raw: assert not (job / 'graph.bin.rawvecs').exists()
                dump(job / 'artifact_savings.json', dict(before=before,
                    after={p.name: dict(logical=p.stat().st_size, allocated=p.stat().st_blocks * 512)
                           for p in job.glob('graph.bin*') if p.is_file()},
                    raw_export_omitted=a.no_raw, cleanup_before_shuffle=True))
        if not (a.resume and (job / 'shuffle.json').exists() and json.loads((job / 'shuffle.json').read_text())['returncode']==0):
            execute(job, 'shuffle', [bindir / 'alliance_shuffle_disk', '--graph_file', job / 'graph.bin',
            '--meta_file', job / 'meta.bin', '--out_graph_file', job / 'ml_graph.bin',
            '--out_meta_file', job / 'shuffle.meta.bin', '--method', 'block_bnf', '--multi_level',
            '--level_k', 2, '--level_m', 2, '--shuffle_o', 2, '--block_size', 4096,
            '--max_iterations', 32,
            '--shuffle_threads', 32, '--shuffle_mem_gb', gb,
            '--space', 'ip'])
        if (bindir / 'alliance_verify_layout').exists():
            execute(job, 'full_readback', [bindir / 'alliance_verify_layout', '--graph', job / 'graph.bin',
                '--meta', job / 'meta.bin', '--ml-graph', job / 'ml_graph.bin',
                '--layout', job / 'ml_graph.bin.layout', '--base', job / 'base.fbin',
                '--format', 'bin', '--attrs', job / 'attrs.i32', '--space', 'ip'])
        dump(job / 'build_shuffle_complete.json', dict(status='passed', synthetic=True,
             geometry=dict(n=4096, dim=dim, wp=wp, degree=32), build_budget_gib=gb))
    dump(root / 'completion.json', dict(status='passed', profiles=['arxiv2m', 'laion10m'],
         scope='synthetic build and shuffle; search is separately validated'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bin', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--no-raw', action='store_true'); p.add_argument('--resume', action='store_true')
    p.add_argument('--finalizer', type=Path)
    run(p.parse_args())
