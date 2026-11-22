"""Read-only compatibility parser for archived navigation-era logs. No execution API."""
import csv
import json
import math
import re
import statistics
from .metrics import aggregate_metric


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cache_policies(c):
    return c.get('cache_policies', {'default': {w:c['parameters'].get('ml_meta_file', '') for w in c['workloads']}})


def parse_log(text, c):
    """Require complete, validated sweeps; never publish partial/failed results."""
    expected = str(c['quantization_fingerprint'])
    require(re.findall(r'^# frozen_quantization: (\d+)$', text, re.M) == [expected], 'initial quantization mismatch')
    require(re.findall(r'^# frozen_quantization_end: (\d+)$', text, re.M) == [expected], 'final quantization mismatch')
    require('# navigation_config: file=' + c['parameters']['navigation_file'] in text, 'navigation not loaded')
    require('error_bound=1 query_estimator=float_rabitq_native float_refine=1' in text, 'unexpected rerank estimator')
    require(text.rstrip().endswith('search done'), 'incomplete native search')
    rows, validated, measured = {}, set(), set()
    header = None; current = None
    for line in text.splitlines():
        if line.startswith('# final_case: ef='):
            current = int(line.split('=')[1])
        elif line.startswith('efs,recall,qps,'):
            header = next(csv.reader([line]))
        elif current is not None and re.match(r'^\d+,', line):
            require(header is not None, 'metrics without header')
            values = next(csv.reader([line]))
            require(len(values) == len(header), 'truncated CSV row')
            row = dict(zip(header, map(float, values)))
            ef = int(row['efs'])
            require(ef == row['efs'] and ef == current and ef not in rows, 'unexpected/duplicate EF row')
            require(all(math.isfinite(v) for v in row.values()), 'nonfinite metric')
            require(0 <= row['recall'] <= 1 and row['qps'] > 0, 'invalid recall/QPS')
            require(row['ml_block_reads'] >= 0 and row['rerank_io_submits'] >= 0, 'negative I/O')
            row['graph_io'] = row['ml_block_reads']
            row['rerank_io'] = row['rerank_io_submits']
            row['total_io'] = row['graph_io'] + row['rerank_io']
            rows[ef] = row
        elif line.startswith('# final_validation: '):
            value = json.loads(line.split(': ', 1)[1])
            require(set(value) == {'invalid', 'duplicates', 'predicate', 'short'} and not any(value.values()),
                    'result validation failed: ' + str(value))
            require(current in rows and current not in validated, 'validation without unique metrics')
            validated.add(current)
        elif line.startswith('# measured_recall: '):
            require(current in rows and current not in measured, 'recall without unique metrics')
            recall = float(line.split(': ', 1)[1])
            require(math.isfinite(recall) and 0 <= recall <= 1 and abs(recall-rows[current]['recall']) < 0.0001,
                    'precise recall mismatch')
            rows[current]['recall'] = recall
            measured.add(current)
    require(set(rows) == validated == measured == set(c['efs']), 'missing or extra EF measurements')
    return [rows[e] for e in c['efs']]


def summarize(rows, c):
    result = []
    for policy, w in ((p,w) for p in cache_policies(c) for w in c['workloads']):
        for t in c['threads']:
            for ef in c['efs']:
                group = [r for r in rows if r.get('cache_policy', 'default') == policy and r['workload'] == w and r['threads'] == t and r['efs'] == ef]
                require(len(group) == c['repeats'] and {r['repeat'] for r in group} == set(range(1, c['repeats']+1)),
                        'incomplete repeat group')
                summary = {'cache_policy': policy, 'workload': w, 'threads': t, 'efs': ef, 'repeats': len(group), 'aggregate': c['aggregate'],
                           'retained_repeats': len(group)-2 if c['aggregate'] == 'trimmed_mean' else len(group)}
                for metric in ('recall', 'qps', 'graph_io', 'rerank_io', 'total_io'):
                    values = [r[metric] for r in group]
                    summary[metric] = aggregate_metric(values, c['aggregate'])
                    summary[metric+'_mean'] = statistics.mean(values)
                    summary[metric+'_stddev'] = statistics.stdev(values) if len(values) > 1 else 0.0
                result.append(summary)
    return result
