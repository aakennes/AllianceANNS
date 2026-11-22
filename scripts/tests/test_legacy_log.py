"""Archived log parser and aggregation compatibility. No execution capability."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchlib import legacy_log as b


def config():
    return dict(efs=[7,20], quantization_fingerprint='123', workloads={'p1': {'query_rng':'/range','gt_file':'/gt'}},
                parameters={'navigation_file':'/navigation','per_query_cache':True,'rerank_policy':'error_bound'},
                threads=[1], cpu_sets={'1':[3]}, binary='/frozen', repeats=3, aggregate='median',
                memory_max_bytes=1073741824, timeout_seconds=1, environment={}, summary_recall_stop=.995)


def log():
    parts = ['First query GT:', '123,456,789,', '# frozen_quantization: 123',
             '# navigation_config: file=/navigation bytes=100',
             '# rerank_config: inline=0 error_bound=1 query_estimator=float_rabitq_native float_refine=1']
    for ef in (7,20):
        parts += [f'# final_case: ef={ef}', 'efs,recall,qps,ml_block_reads,rerank_io_submits',
                  'efs,recall,qps,ml_block_reads,rerank_io_submits', f'{ef},0.75,100,2,3',
                  '# final_validation: {"invalid":0,"duplicates":0,"predicate":0,"short":0}',
                  '# measured_recall: 0.75000000001']
    return '\n'.join(parts + ['# frozen_quantization_end: 123', 'search done'])


class BenchmarkTests(unittest.TestCase):
    def test_native_headers_gt_preamble_and_precision(self):
        rows = b.parse_log(log(), config())
        self.assertEqual([r['efs'] for r in rows], [7,20])
        self.assertEqual(rows[0]['total_io'], 5)
        self.assertEqual(rows[0]['recall'], .75000000001)

    def test_reject_bad_or_incomplete_output(self):
        base = log()
        cases = [base.replace('search done',''), base.replace('end: 123','end: 124'),
                 base.replace('"invalid":0','"invalid":1'),base.replace('"short":0','"short":1'),
                 base.replace('0.75,100','0.75,nan'),base.replace('file=/navigation','file=/wrong'),
                 base.replace('# measured_recall: 0.75000000001','# measured_recall: 0.2'),
                 base.replace('20,0.75,100,2,3',''),base.replace('7,0.75,100,2,3','7,0.75,100,2,3\n7,0.75,100,2,3')]
        for text in cases:
            with self.subTest(text=text[-100:]), self.assertRaises(ValueError): b.parse_log(text, config())

    def test_median_not_best_run_and_sum_before_aggregation(self):
        rows=[]
        for r, qps, graph, rerank in [(1,100,1,100),(2,300,10,1),(3,200,100,10)]:
            for ef in (7,20):
                rows.append(dict(workload='p1',threads=1,repeat=r,efs=ef,recall=.9,qps=qps,
                                 graph_io=graph,rerank_io=rerank,total_io=graph+rerank))
        s=b.summarize(rows,config())[0]
        self.assertEqual(s['qps'],200)
        self.assertEqual(s['total_io'],101)  # not median(graph)+median(rerank)=20
        self.assertEqual(s['qps_stddev'],100)
        c=config();c['aggregate']='mean'
        self.assertEqual(b.summarize(rows,c)[0]['total_io'],74)
        with self.assertRaises(ValueError): b.summarize(rows[:-1],config())

    def test_trimmed_mean_and_cache_policy_isolation(self):
        c=config();c.update(repeats=5,aggregate='trimmed_mean')
        c['cache_policies']={'equal':{'p1':'/equal'},'landing':{'p1':'/landing'}}
        rows=[]
        for policy,offset in [('equal',0),('landing',100)]:
            for r,value in enumerate([1,2,4,9,1000],1):
                for ef in c['efs']:
                    rows.append(dict(cache_policy=policy,workload='p1',threads=1,repeat=r,efs=ef,
                        recall=.9,qps=value+offset,graph_io=value,rerank_io=2,total_io=value+2))
        s=b.summarize(rows,c)
        self.assertEqual(s[0]['qps'],5)
        self.assertEqual(s[2]['qps'],105)
        self.assertEqual(s[0]['retained_repeats'],3)
        self.assertEqual(s[0]['total_io'],7)
        # Tied extrema remove exactly one occurrence at each end.
        for row in rows: row['qps']=7
        self.assertEqual(b.summarize(rows,c)[0]['qps'],7)

if __name__ == '__main__': unittest.main()
