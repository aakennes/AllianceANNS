"""Search only the synthetic acceptance fixtures and independently replay results."""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import struct
import subprocess
import time
import numpy as np

GIB = 1024**3
MIB = 1024**2


def dump(p, data):
    p.write_text(json.dumps(data, indent=2) + '\n')


def call(root, name, command, env=None, scope=None, cap=None):
    cmd = list(map(str, command))
    if scope:
        cmd = ['systemd-run', '--user', '--scope', '--quiet', '--unit=' + scope,
               '-p', 'MemoryMax=' + str(cap), '-p', 'MemorySwapMax=0',
               'taskset', '-c', '1'] + cmd
    start = time.monotonic()
    with (root / (name + '.log')).open('w') as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                           cwd=str(root), timeout=600)
    dump(root / (name + '.command.json'), dict(command=cmd, returncode=p.returncode,
         elapsed_seconds=time.monotonic() - start))
    if p.returncode:
        raise RuntimeError(name + ': ' + (root / (name + '.log')).read_text()[-3500:])


def read_rows(path, dtype):
    data = path.read_bytes(); pos = 0; rows = []
    while pos < len(data):
        d, = struct.unpack_from('<I', data, pos); pos += 4
        rows.append(np.frombuffer(data, dtype=dtype, count=d, offset=pos).copy()); pos += 4*d
    assert pos == len(data)
    return np.array(rows)


def replay(job, out):
    n, d = struct.unpack('<II', (job / 'base.fbin').read_bytes()[:8])
    base = np.memmap(job / 'base.fbin', dtype='<f4', mode='r', offset=8, shape=(n,d))
    attrs = np.fromfile(job / 'attrs.i32', dtype='<i4')
    q = read_rows(job / 'query.fvecs', '<f4')
    gt = read_rows(job / 'gt.ivecs', '<i4')
    ranges = np.fromfile(job / 'ranges.bin', dtype='<i4').reshape(-1,2)
    report = []
    for p in sorted(out.glob('r*/ef*/measured.json')):
        m = json.loads(p.read_text()); point = p.parent
        raw = (point / 'results.bin').read_bytes()
        magic, nq, k, ef = struct.unpack_from('<4Q', raw)
        assert magic == 0x414c4c49414e4331 and nq == len(q) and k == 10 and ef == m['ef']
        assert len(raw) == 32 + nq*k*12
        hits = 0; pos = 32
        for i in range(nq):
            seen = set()
            for _ in range(k):
                label, distance = struct.unpack_from('<Qf', raw, pos); pos += 12
                assert label < n and label not in seen and math.isfinite(distance)
                seen.add(label)
                assert ranges[i,0] <= attrs[label] <= ranges[i,1]
                exact = 1.0 - float(np.dot(base[label].astype('f8'), q[i].astype('f8')))
                assert abs(distance-exact) <= 2e-5, (label, distance, exact)
            hits += len(seen.intersection(gt[i].tolist()))
        assert hits == m['matches'] and hits / (nq*k) == m['recall']
        assert abs(nq/m['batch_wall_seconds'] - m['qps']) < 1e-9
        per = list(csv.DictReader((point / 'queries.csv').open()))
        assert len(per) == nq and [int(x['query_id']) for x in per] == list(range(nq))
        for col, key in [('requests','io_requests'),('completions','io_completions'),('bytes','io_bytes'),
                         ('graph_io','graph_io'),('rerank_io','rerank_io'),('matches','matches')]:
            assert sum(int(x[col]) for x in per) == m[key]
        assert m['io_requests'] == m['io_completions'] == m['graph_io'] + m['rerank_io']
        assert m['io_bytes'] == m['edge_block_bytes']*m['graph_io'] + m['vector_block_bytes']*m['rerank_io']
        assert m['search_phase_peak_rss_bytes'] <= m['search_rss_limit_bytes']
        for key in ['invalid','duplicates','predicate','short','observer_errors','unknown_fd','bad_layout','pending']:
            assert m[key] == 0
        report.append(dict(ef=ef, queries=nq, recall=m['recall'], qps=m['qps'],
             logical_reads=m['io_requests']/nq, io_4k_equivalent=m['io_bytes']/4096/nq,
             search_rss_bytes=m['search_phase_peak_rss_bytes'], static_resident_bytes=m['static_resident_bytes']))
    assert len(report) == 3
    return report


def run(a):
    bindir, root = a.bin.resolve(), a.fixture.resolve()
    reports = {}
    for name, dim, wp, nc, static, memory, warmup, per_query, auto in [
            ('arxiv2m',384,5,256,480*MIB,'whole_process',1,1,1),
            ('laion10m',512,6,128,196*MIB,'phase',0,0,0)]:
        job = root/name; output = job/a.tag
        assert not output.exists(), 'do not overwrite measured fixture points'
        mode = 'auto' if name == 'arxiv2m' else 'graph'
        # Sidecar preparation uses structural routing; no ANN warmup queries.
        landing = job/('landing_'+a.tag+'.bin')
        call(job, 'landing_'+a.tag, [bindir/'alliance_collect_landing', '--graph_file',job/'graph.bin',
             '--meta_file',job/'meta.bin','--query_rng',job/'ranges.bin','--out_file',landing,
             '--report_file',job/('landing_'+a.tag+'.json'),'--space','ip','--filter_mode',mode])
        prefix=job/('static_'+a.tag)
        call(job,'static_'+a.tag,[bindir/'alliance_prepare_static_cache','--graph_file',job/'graph.bin',
             '--meta_file',job/'meta.bin','--ml_graph_file',job/'ml_graph.bin','--ml_meta_file',job/'ml_graph.bin.layout',
             '--out_prefix',prefix,'--budgets',static,'--space','ip','--static_landing_file',landing])
        opts=[bindir/'alliance_benchmark','--graph_file',job/'graph.bin','--meta_file',job/'meta.bin',
             '--ml_graph_file',job/'ml_graph.bin','--ml_meta_file',str(prefix)+'_'+str(static)+'.layout',
             '--query_vec',job/'query.fvecs','--query_rng',job/'ranges.bin','--gt_file',job/'gt.ivecs',
             '--space','ip','--rabitq_bits',3,'--packed_rabitq',1,'--pipeline',1,'--threads',1,
             '--k',10,'--beam_width',8,'--rerank_policy','error_bound',
             '--ml_cache_bytes',8*MIB,'--ml_cache_policy','fifo','--ml_cache_auto',auto,
             '--filter_mode',mode,'--prefilter_threshold',8192,'--top_layer_postfilter',1 if name=='arxiv2m' else 0,
             '--rerank_beam_width',64 if name=='arxiv2m' else 8,
             '--rerank_batch_blocks',64 if name=='arxiv2m' else 8,'--rerank_candidate_factor',2,
             '--rerank_bound_prescreen',1,'--efs_list','16,64,256']
        if per_query: opts += ['--per_query_cache']
        opts += ['--entry_sampling','guided','--entry_samples',256,'--entry_count',4]
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        if name=='laion10m': env['MALLOC_ARENA_MAX']='4'
        else: env.pop('MALLOC_ARENA_MAX',None)
        scope='alliance-bench-round3-'+name+'-'+a.tag
        for key,value in dict(OUTPUT=output,EXPECTED_N=4096,EXPECTED_D=dim,EXPECTED_WP=wp,EXPECTED_M=32,
             EXPECTED_MAX_LAYER=wp,EXPECTED_CENTROIDS=nc,EXPECTED_QUERIES=32,EXPECTED_BITS=3,
             QUANTIZATION_SOURCE='meta',STATIC_BUDGET=static,DYNAMIC_BUDGET=8*MIB,
             LOADING_RSS_BYTES=GIB if memory=='whole_process' else 9*GIB,
             SEARCH_RSS_BYTES=GIB if memory=='whole_process' else 3*GIB,
             MEMORY_MODE=memory,SCOPE_PREFIX='alliance-bench-',REPEATS=1,WARMUPS=warmup,
             ORDER='ef_then_repeat' if name=='arxiv2m' else 'repeat_then_ef').items():
            env['BENCH_'+key]=str(value)
        fp=job/('fingerprint_'+a.tag+'.json')
        env['BENCH_MODE']='fingerprint'; env['BENCH_OUTPUT']=str(fp)
        call(job,'fingerprint_'+a.tag,opts,env)
        env['BENCH_MODE']='measure';env['BENCH_OUTPUT']=str(output)
        env['BENCH_EXPECTED_FP']=json.loads(fp.read_text())['fingerprint']
        dump(job/('search_protocol_'+a.tag+'.json'),dict(environment={k:v for k,v in env.items() if k.startswith('BENCH_')},
             synthetic=True,production_timing_comparison=False,
             entry_policy='guided',historical_navigation_equivalent=False))
        call(job,'search_'+a.tag,opts,env,scope,GIB if memory=='whole_process' else 9*GIB)
        reports[name]=replay(job,output)
    dump(root/('search_verification_'+a.tag+'.json'),dict(status='passed',synthetic=True,profiles=reports,
         correctness='raw archived IDs/distances, predicates and GT replayed outside measured search',
         scope='small-scale compatibility; QPS does not represent the production datasets'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bin',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--no-raw',action='store_true');run(p.parse_args())
