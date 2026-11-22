#!/usr/bin/env python3
"""Shared JSON reader and runner for build.sh, shuffle.sh and search.sh.

Commands are argv arrays, never shell-evaluated. Dry runs create no outputs.
Existing indexes/results are never overwritten. No automatic retries or jobs.
"""
import argparse
from copy import deepcopy
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchlib.metrics import aggregate_metric
from benchlib.build_artifacts import DiskObservation, finalization_argv
from benchlib.shuffle_memory import scope_memory_properties, ShuffleRssObservation, rss_measurements
from validate_shuffle_completion import validate_completion


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merge(base, override):
    """Recursively merge dictionaries; an override list replaces the entire list."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(directory, stage, workload, defaults_path=None):
    """Read the selected AllianceANN stage; query and ef files are independent."""
    directory = Path(directory)
    require(directory.is_dir(), f'Expected a dataset config DIRECTORY: {directory}')
    script_root = Path(__file__).resolve().parent
    defaults_path = Path(defaults_path) if defaults_path else script_root / 'defaults.json'
    sources = []
    def source(path):
        value = read(path)
        sources.append({'path': str(Path(path).resolve()), 'sha256': sha(path)})
        return value
    defaults = source(defaults_path)
    dataset = source(directory / 'dataset.json')
    stage_override = source(directory / (stage + '.json'))
    section = merge(defaults[stage], stage_override)
    section.setdefault('memory_max_bytes', dataset['memory_max_bytes'])
    config = {'dataset': dataset['dataset'], 'paths': merge(defaults['paths'], dataset['paths']),
              'environment': merge(defaults['environment'], dataset.get('environment', {}))}
    if 'index' in dataset:
        config['index'] = dataset['index']
    if stage == 'build':
        config['build'] = {'allianceann': section}
    else:
        query_config = source(directory / 'queries.json')
        require(workload in query_config['workloads'], f'Workload {workload} is not configured for {dataset["dataset"]}')
        query = merge(query_config.get('defaults', {}), query_config['workloads'][workload])
        protocol = merge(defaults['search_protocol'], section.pop('protocol', {}))
        protocol['queries'] = {workload: {'allianceann': query}}
        if stage == 'shuffle':
            config['shuffle'] = section
        else:
            profiles = source(directory / 'efs.json')
            protocol['ef_profiles'] = {f'allianceann/{w}/k{k}': values
                for w, choices in profiles.items() for k, values in choices.items()}
            section['parameters'] = merge(section['parameters'], dataset['index'])
            section['entry_sampling_explicit'] = 'entry_sampling' in stage_override.get('parameters', {})
            protocol['allianceann'] = section
        config['search'] = protocol
    config['config_sources'] = sources
    return config


def flag_args(parameters):
    args = []
    for key, value in parameters.items():
        if value is False or value is None:
            continue
        args.append('--' + key)
        if value is not True:
            args.append(str(value))
    return args


def existing(*paths):
    for path in paths:
        require(Path(path).is_file(), f'Missing input: {path}')


def check_binary(section):
    path = section['binary']
    existing(path)
    require(os.access(path, os.X_OK), f'Not executable: {path}')
    if section.get('sha256'):
        require(sha(path) == section['sha256'], f'Binary changed: {path}; update JSON after reviewing the new build')


def clean_environment(config, threads):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PIPE_', 'FINAL_', 'TCMALLOC_', 'MALLOC_', 'BENCH_',
                                'ALLIANCE_ABLATION_', 'ALLIANCE_WOW_', 'ALLIANCE_TRACE'))}
    env.update(config['environment'], OMP_NUM_THREADS=str(threads))
    return env


def run_command(argv, name, context, memory, extra_env=None):
    args, config, out = context
    env = clean_environment(config, context.threads)
    env.update({k: str(v) for k, v in (extra_env or {}).items()})
    unit = f'alliance-script-{name.replace("_", "-")}-{time.time_ns()}'
    require(memory is not None or name == 'shuffle', 'Unlimited memory is supported only for native shuffle')
    command = ['systemd-run', '--user', '--scope', '--quiet', '--unit=' + unit] + scope_memory_properties(memory) + [
               '/usr/bin/time', '-v', '-o', str(out / (name + '.time.txt'))] + list(map(str, argv))
    print(shlex.join(command), flush=True)
    if args.dry_run:
        if extra_env:
            print('  env: ' + json.dumps(extra_env, ensure_ascii=False), flush=True)
        return '', 0
    save(out / (name + '.command.json'), {'argv': command, 'environment': {
        k: v for k, v in env.items() if k in config['environment'] or k.startswith(
            ('PIPE_', 'FINAL_', 'OMP_', 'TCMALLOC_', 'ALLIANCE_ABLATION_', 'ALLIANCE_WOW_'))}})
    start = time.monotonic()
    # Observe file lifetimes only during construction; search timing is unchanged.
    observe_disk = name.startswith(('build_', 'finalize_', 'shuffle'))
    observer = DiskObservation(out, name) if observe_disk else None
    with (out / (name + '.log')).open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        rss_observer = ShuffleRssObservation(process.pid) if name == 'shuffle' else None
        next_disk = 0
        while observe_disk and process.poll() is None:
            elapsed = time.monotonic() - start
            if elapsed >= next_disk:
                observer.sample(elapsed)
                next_disk = elapsed + 1
            if rss_observer:
                rss_observer.sample()
            try:
                process.wait(timeout=0.1 if rss_observer else 1)
            except subprocess.TimeoutExpired:
                pass
        code = process.wait()
    if observe_disk:
        observer.sample(time.monotonic() - start)
        save(out / (name + '.disk.json'), observer.close())
    text = (out / (name + '.log')).read_text(errors='replace')
    require(code == 0, f'{name}: exit={code}; see {out / (name + ".log")}\n{text[-1500:]}')
    timing = (out / (name + '.time.txt')).read_text()
    match = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', timing)
    require(match is not None, f'{name}: missing RSS measurement')
    rss = int(match[1]) / 1024
    observation = rss_measurements(text, rss, rss_observer) if rss_observer else {'peak_rss_mib': rss}
    rss = observation['peak_rss_mib']
    if memory is not None:
        require(rss * (1 << 20) <= memory, f'{name}: RSS exceeds configured cap')
    save(out / (name + '.metrics.json'), {'seconds': time.monotonic() - start,
         **observation, 'memory_max_bytes': memory, 'returncode': code})
    return text, rss


class Context(tuple):
    # Keep the command runner's inputs together without module-global run state.
    pass


def raw_index(config, directory=None):
    if directory:
        manifest = read(Path(directory) / 'index.json')
        require(manifest.get('algorithm') == 'allianceann', 'Expected an AllianceANN index manifest')
        require(manifest.get('dataset') == config['dataset'], 'Index dataset mismatch')
        return manifest['graph_file'], manifest['meta_file']
    params = config['index']
    return params['graph_file'], params['meta_file']


def build(context):
    args, config, out = context
    section = config['build'][args.algorithm]
    check_binary(section)
    context.threads = int(section.get('threads', section.get('parameters', {}).get('threads', 32)))
    params = dict(section['parameters'])
    existing(params['basevec'], params['baseatt'])
    params.update(graph_file=str(out / 'graph.bin'), meta_file=str(out / 'meta.bin'),
                  checkpoint_meta=str(out / 'checkpoint.bin'))
    upper = section.get('upper_wow')
    if upper:
        check_binary(upper)
        params['meta_file'] = str(out / 'seed_meta.bin')
    run_command([section['binary']] + flag_args(params), 'build_alliance', context,
                section['memory_max_bytes'])
    if upper:
        env = dict(upper['environment'], ALLIANCE_ABLATION_CLUSTERS=str(out / 'clusters.bin'))
        run_command([upper['binary'], params['graph_file'], params['meta_file'], out / 'meta.bin',
                     'wow', upper['efc'], upper['passes'], config['dataset'], upper['K']],
                    'build_upper_wow', context, section['memory_max_bytes'], env)
    # Close resume only after the existing full artifact verifier succeeds.
    # Failed builds retain vectors and checkpoints for recovery.
    resume_closed = False
    if section.get('finalize_on_success', False):
        require(not upper, 'automatic finalization requires the native completed build path')
        verifier = section['verifier_binary']
        existing(verifier)
        argv = finalization_argv(sys.executable, Path(__file__).with_name('finalize_build.py'),
                                 verifier, out, params)
        run_command(argv, 'finalize_build', context, section['memory_max_bytes'])
        resume_closed = True
    manifest = {'dataset': config['dataset'], 'algorithm': args.algorithm,
                'graph_file': params['graph_file'], 'meta_file': str(out / 'meta.bin'),
                'upper_builder': section['upper_builder'],
                'cross_cluster_inherit': section.get('cross_cluster_inherit'),
                'build_parameters': params, 'resume_closed': resume_closed,
                'rawvec_export': params.get('rawvec_export', 'on')}
    if not args.dry_run:
        save(out / 'index.json', manifest)


def shuffle(context):
    args, config, out = context
    section = config['shuffle']
    # An unlimited native shuffle must not silently uncap cache preparation.
    cache_memory = section.get('cache_memory_max_bytes', section['memory_max_bytes'])
    require(cache_memory is not None, 'Set cache_memory_max_bytes when shuffle memory_max_bytes is null')
    check_binary(section)
    graph, meta = raw_index(config, args.index_dir)
    existing(graph, meta)
    context.threads = int(section['parameters'].get('shuffle_threads', 32))
    params = dict(section['parameters'], graph_file=graph, meta_file=meta,
                  out_graph_file=str(out / 'ml_graph.bin'), out_meta_file=str(out / 'ml_meta.bin'))
    run_command([section['binary']] + flag_args(params), 'shuffle', context, section['memory_max_bytes'])
    if not args.dry_run:
        validate_completion(out / 'ml_graph.bin', out / 'ml_graph.bin.layout', out / 'ml_meta.bin')
    query = config['search']['queries'][args.workload]['allianceann']
    existing(section['collect_binary'], section['cache_binary'], query['query_rng'])
    landing = out / 'landing.bin'
    run_command([section['collect_binary'], '--graph_file', graph, '--meta_file', meta,
                 '--query_rng', query['query_rng'], '--out_file', landing,
                 '--report_file', out / 'landing.json', '--space', params['space'],
                 '--filter_mode', 'auto', '--prefilter_threshold', 8192, '--query_limit', 0,
                 '--can_prefilter', 'true'], 'collect_landing', context, cache_memory)
    budget = section['static_budget_bytes']
    run_command([section['cache_binary'], '--graph_file', graph, '--meta_file', meta,
                 '--ml_graph_file', out / 'ml_graph.bin', '--ml_meta_file', out / 'ml_graph.bin.layout',
                 '--out_prefix', out / 'static', '--budgets', budget,
                 '--static_landing_file', landing, '--space', params['space']],
                'static_cache', context, cache_memory)
    if not args.dry_run:
        save(out / 'index.json', {'dataset': config['dataset'], 'algorithm': args.algorithm,
             'graph_file': graph, 'meta_file': str(out / 'ml_meta.bin'), 'ml_graph_file': str(out / 'ml_graph.bin'),
             'static_layout': str(out / f'static_{budget}.layout'), 'cache_workload': args.workload,
             'ml_only': True, 'shuffle_completion': str(out / 'ml_graph.bin') + '.completion.json'})


def parse_alliance(text, section):
    rows, current, header = [], None, None
    for line in text.splitlines():
        if line.startswith('# final_case:'):
            current = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
        elif line.startswith('efs,recall,qps,'):
            header = line.split(',')
        elif current is not None and header and re.match(r'^\d+,', line):
            current.update(zip(header, map(float, line.split(','))))
        elif current is not None and line.startswith('# latency:'):
            current.update((k, float(v)) for k, v in re.findall(r'(\w+)=([-+\d.eE]+)', line))
        elif current is not None and line.startswith('# final_routing:'):
            current.update(json.loads(line.split(': ', 1)[1]))
        elif current is not None and line.startswith('# final_validation:'):
            current.update(json.loads(line.split(': ', 1)[1])); rows.append(current)
            current = None; header = None
        elif line.startswith('# measured_recall:'):
            rows[-1]['unique_recall'] = float(line.split(': ', 1)[1])
    hashes = re.findall(r'^# frozen_quantization(?:_end)?: (\d+)$', text, re.M)
    require(hashes == [section['fingerprint']] * 2, 'Frozen quantization fingerprint mismatch')
    require('bitwise_equal=1' in text, 'Missing packed bitwise validation')
    for row in rows:
        require(all(row.get(k) == 0 for k in ['invalid', 'duplicates', 'predicate']), 'Invalid search output')
        row['total_io'] = row['io_submits'] + row['rerank_io_submits']
    return rows


def resolve_navigation(params, manifest=None, index_dir=None, entry_sampling_explicit=False):
    """Bind navigation to the selected index, never a previous config's graph."""
    if manifest is not None:
        # --index-dir replaces the configured index as a unit. A navigation
        # override from the old dataset config must not leak into the new index.
        params.pop('navigation_file', None)
    sampling = params.get('entry_sampling')
    eligible = (int(params.get('pipeline', 1)) == 1 and params.get('ml_graph_file') and
                params.get('filter_mode', 'auto') in ('graph', 'auto'))
    if not eligible:
        require(not params.get('navigation_file'), 'Navigation requires ML graph/auto search')
        if sampling == 'navigation':
            require(not entry_sampling_explicit,
                    'Explicit entry_sampling=navigation requires ML graph/auto search')
            params['entry_sampling'] = 'guided'
        return
    if sampling not in (None, 'navigation') and not params.get('navigation_file'):
        return  # Explicit guided/global/range retains the requested entry policy.
    navigation = params.get('navigation_file')
    if manifest is not None and manifest.get('navigation_file'):
        navigation = Path(manifest['navigation_file'])
        if not navigation.is_absolute():
            navigation = Path(index_dir) / navigation
    if not navigation:
        graph = params.get('graph_file') or (params.get('ml_graph_file') if params.get('ml_only') else None)
        require(graph, 'Navigation requires graph_file or an ML-only ml_graph_file')
        navigation = Path(graph).parent / 'navigation.bin'
    navigation = Path(navigation).resolve()
    require(navigation.is_file(),
            f'Missing navigation asset for the selected index: {navigation}; '
            'provide its compatible navigation_file or explicitly select entry_sampling=guided; '
            'do not reuse another graph\'s navigation file')
    params['navigation_file'] = str(navigation)
    params['entry_sampling'] = 'navigation'


def search(context):
    args, config, out = context
    section = config['search'][args.algorithm]
    settings = config['search']; queries = settings['queries'][args.workload][args.algorithm]
    check_binary(section)
    context.threads = settings['threads']
    require(context.threads == 16, 'This frozen protocol is configured for16 threads; prepare a separate validated profile for other counts')
    efs = args.efs or settings['ef_profiles'].get(
        f'{args.algorithm}/{args.workload}/k{args.k}', settings['fallback_efs'][str(args.k)])
    require(efs == sorted(set(efs)) and all(v > 0 and v % 5 == 0 for v in efs),
            'ef must be strictly increasing positive multiples of5; ef<k is allowed')
    require(efs[-1] <= 64 * args.k, 'Pinned search drivers support ef<=64*k')
    env = {}
    params = dict(section['parameters'], query_vec=queries['query_vec'], query_rng=queries['query_rng'],
                  gt_file=queries['gt'][str(args.k)], k=args.k, efs=efs[0], ml_meta_file=queries['layout'])
    manifest = None
    if args.index_dir:
        manifest = read(Path(args.index_dir) / 'index.json')
        require(manifest.get('algorithm') == args.algorithm and manifest.get('dataset') == config['dataset'],
                'Index manifest algorithm/dataset mismatch')
        for key in ['meta_file', 'ml_graph_file']:
            params[key] = manifest[key]
        params.pop('graph_file', None)
        if manifest.get('graph_file'):
            params['graph_file'] = manifest['graph_file']
        params['ml_meta_file'] = manifest['static_layout']
        if manifest.get('ml_only', False):
            require(section.get('driver') == 'main', 'ML-only artifacts require the native search driver')
            params['ml_only'] = True
        require(manifest['cache_workload'] == args.workload, 'Prepare static cache for this workload first')
    if section.get('driver') == 'main':
        resolve_navigation(params, manifest, args.index_dir,
                           entry_sampling_explicit=section.get('entry_sampling_explicit', False))
    required = ['meta_file', 'ml_graph_file', 'ml_meta_file',
                'query_vec', 'query_rng', 'gt_file', 'rabitq_centroids', 'rabitq_cids']
    if not params.get('ml_only', False):
        required.append('graph_file')
    existing(*(params[k] for k in required), section['quant_checkpoint'])
    if section.get('driver') == 'main':
        params.pop('efs', None)
        params.update(quant_checkpoint=section['quant_checkpoint'],
                      efs_list=','.join(map(str, efs)), recall_stop=settings['recall_stop'])
    else:
        env.update(FINAL_QUANT_CHECKPOINT=section['quant_checkpoint'], FINAL_EFS=','.join(map(str, efs)),
                   FINAL_RECALL_STOP=str(settings['recall_stop']))
    argv = [section['binary']] + flag_args(params)
    runs = []
    for repeat in range(1, args.repeats + 1):
        text, rss = run_command(['taskset', '-c', '1-16'] + argv, f'search_r{repeat}', context,
                                section['memory_max_bytes'], env)
        if args.dry_run:
            continue
        rows = parse_alliance(text, section)
        require(rows and [int(v['efs']) for v in rows] == efs[:len(rows)], 'Unexpected ef output sequence')
        require(len(rows) == len(efs) or rows[-1]['unique_recall'] >= settings['recall_stop'],
                'Search stopped before the recall threshold')
        require(all(v['unique_recall'] < settings['recall_stop'] for v in rows[:-1]),
                'A larger ef was tested after the threshold crossing')
        save(out / f'search_r{repeat}.json', {'values': rows, 'peak_rss_mib': rss})
        runs.append((rows, rss))
    if args.dry_run:
        return
    fields = ['ef', 'recall', 'qps', 'io_times', 'graph_io', 'rerank_io', 'rss_mib',
              'short_queries_max', 'repeats', 'threads', 'k']
    result = []
    for ef in efs:
        measured = [(row, rss) for rows, rss in runs for row in rows if int(row['efs']) == ef]
        if not measured:
            continue
        med = lambda key: aggregate_metric((row[key] for row, _ in measured), 'median')
        result.append(dict(ef=ef, recall=med('unique_recall'), qps=med('qps'),
            io_times=med('total_io'),
            graph_io=med('io_submits'), rerank_io=med('rerank_io_submits'),
            rss_mib=aggregate_metric((rss for _, rss in measured), 'median'),
            short_queries_max=max(row.get('short', row.get('short_queries', 0)) for row, _ in measured),
            repeats=len(measured), threads=context.threads, k=args.k))
    root = Path(config['paths']['result_root']) / args.algorithm / config['dataset']
    csv_path = root / f'k{args.k}' / f'{config["dataset"]}_{args.workload}_k{args.k}_3bit_{out.name}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(result)
    save(root / 'json' / (csv_path.stem + '.provenance.json'), {
        'csv': str(csv_path), 'sha256': sha(csv_path), 'run_dir': str(out), 'config_dir': str(args.config),
        'config_sources': config.get('config_sources', []),
        'requested_efs': efs, 'recall_stop': settings['recall_stop'], 'index': section,
        'rerank_policy': settings['rerank_policy'], 'repeats_requested': args.repeats,
        'aggregation': 'Median per ef; repeats column is the actual count. No extrapolation.',
        'cache_note': 'Static cache uses the configured workload layout; dynamic cache uses the configured search parameters.'})
    print('CSV: ' + str(csv_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'shuffle', 'search'])
    parser.add_argument('dataset', choices=['sift1m', 'arxiv2m', 'deep10m'])
    parser.add_argument('--config-dir', '--config', dest='config', type=Path,
                        help='Dataset config directory; default: scripts/configs/<dataset>/')
    parser.add_argument('--defaults', type=Path, help='Optional replacement defaults.json')
    parser.add_argument('--algorithm', choices=['allianceann'],
                        help='AllianceANN only; the legacy PipeANNFilter dataset entry is retired')
    parser.add_argument('--entry-algorithm', choices=['allianceann'], help=argparse.SUPPRESS)
    parser.add_argument('--out-dir', type=Path, help='New directory for logs/outputs; must not exist')
    parser.add_argument('--index-dir', type=Path, help='Index manifest produced by build/shuffle (AllianceANN)')
    parser.add_argument('--workload', choices=['mixed', 'p1', 'p4', 'p7', 'origin'], default='mixed')
    parser.add_argument('--k', type=int, choices=[10, 25, 100], default=10, help='Query k, not cluster K')
    parser.add_argument('--efs', type=lambda s: [int(v) for v in s.split(',')])
    parser.add_argument('--repeats', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true', help='Validate paths and print commands; create no files')
    args = parser.parse_args()
    args.algorithm = args.algorithm or args.entry_algorithm or 'allianceann'
    script_root = Path(__file__).resolve().parent
    args.config = args.config or script_root / 'configs' / args.dataset
    config = load_config(args.config, args.stage, args.workload, args.defaults)
    require(config['dataset'] == args.dataset, 'JSON dataset mismatch')
    args.repeats = config.get('search', {}).get('repeats', 3) if args.repeats is None else args.repeats
    require(args.repeats > 0, 'repeats must be positive')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    default_root = config['paths']['run_root'] if args.stage == 'search' else config['paths']['index_run_root']
    out = args.out_dir or Path(default_root) / f'{args.stage}_{args.algorithm}_{args.dataset}_{stamp}'
    require(not out.exists(), f'Refusing to overwrite existing run: {out}')
    context = Context((args, config, out)); context.threads = 32
    # Preflight first. Dry-run performs no mutations, including output directory creation.
    preview = argparse.Namespace(**vars(args)); preview.dry_run = True
    check = Context((preview, config, out)); check.threads = context.threads
    handlers = {'build': build, 'shuffle': shuffle, 'search': search}
    if not args.dry_run:
        handlers[args.stage](check)
        out.mkdir(parents=True, exist_ok=False)
        save(out / 'config.json', config)
    try:
        handlers[args.stage](context)
        if not args.dry_run:
            save(out / 'status.json', {'state': 'completed', 'stage': args.stage, 'dataset': args.dataset})
    except Exception as error:
        if not args.dry_run and out.exists():
            save(out / 'status.json', {'state': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        sys.exit('ERROR: ' + str(error))
