"""Verify shared disk-helper contracts without launching production queries."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('shuffle_receipt', ROOT/'scripts/validate_shuffle_completion.py')
receipt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipt)

class DiskScripts(unittest.TestCase):
    def call(self, script, *args, **kw):
        p = subprocess.run(['bash', str(ROOT/'scripts'/script)] + list(args),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True)
        self.assertEqual(p.returncode, kw.get('exit', 0), p.stdout + p.stderr)
        commands = ('alliance_search_disk ', 'alliance_build_disk ')
        if kw.get('all_commands'):
            commands += ('alliance_shuffle_disk ', 'prepare_selectivity_static_cache.py ')
        return [shlex.split(x) for x in p.stdout.splitlines() if any(name in x for name in commands)]

    def search(self, dataset, *args, **kwargs):
        return self.call('disk_search_common.sh', dataset, *args, **kwargs)

    def build(self, dataset, *args, **kwargs):
        return self.call('disk_build_common.sh', dataset, *args, **kwargs)

    def test_shuffle_defaults_and_retired_controls(self):
        rows = self.search('arxiv', '--dry_run',
                         '--skip_search', '--methods', 'block_bnf,identity', all_commands=True)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row[row.index('--max_iterations') + 1], '32')
            self.assertNotIn('--move_margin', row)
            self.assertNotIn('--min_colocatable_for_bnf', row)
            self.assertNotIn('--static_cache_budget', row)
            self.assertNotIn('--static_landing_file', row)
        for args in [('--move_margin', '1'), ('--min_colocatable_for_bnf', '0')]:
            self.search('arxiv', '--dry_run', *args, exit=2)
        for method in ['affinity_bnf', 'weight_sorted', 'in_weight_sorted', 'bfs_pack', 'unknown']:
            self.search('arxiv', '--dry_run', '--methods', method, exit=2)

    def test_skip_search_still_prepares_caches(self):
        for landing in ['false', 'true']:
            rows = self.search('sift1m', '--dry_run',
                             '--skip_search', '--selectivities', '1,4', '--static_cache_budget', '4096',
                             '--static_cache_use_landing', landing, all_commands=True)
            self.assertEqual(len(rows), 3)  # One base layout, two requested cache preparations.
            self.assertTrue(any(Path(item).name == 'alliance_shuffle_disk' for item in rows[0]))
            for row in rows[1:]:
                self.assertTrue(any('prepare_selectivity_static_cache.py' in item for item in row))
                self.assertEqual(row[row.index('--budgets') + 1], '4096')
                self.assertEqual(row[row.index('--static_cache_use_landing') + 1], landing)
                self.assertFalse(any('alliance_search_disk' in item for item in row))

    def test_automatic_reuse_requires_completion_but_explicit_legacy_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, meta = root/'source.bin', root/'meta.bin'
            source.touch(); meta.touch()
            graph = root/'source_ml_o2_k2m2_block_bnf_graph.bin'
            layout = Path(str(graph) + '.layout')
            exported_meta = root/'source_ml_o2_k2m2_block_bnf_meta.bin'
            for p in [graph, layout, exported_meta]: p.write_bytes(b'fixture')
            args = ('--graph_file', str(source), '--meta_file', str(meta), '--skip_search', '--out_dir', directory)
            self.search('arxiv', *args, exit=2)
            self.search('arxiv', *args, '--skip_shuffle')
            write_receipt(graph, layout, exported_meta)
            self.search('arxiv', *args)
            graph.write_bytes(b'changed')  # Same byte count: must compare the SHA, not just size.
            self.search('arxiv', *args, exit=2)

    def test_defaults_and_budgets(self):
        for ds, nc, count in [('arxiv', '256', 5), ('sift1m', '128', 20)]:
            rows = self.search(ds, '--dry_run')
            self.assertEqual(len(rows), count)
            for i, row in enumerate(rows):
                def opt(key): return row[row.index(key)+1]
                self.assertEqual(opt('--rerank_num'), '0')
                self.assertEqual(opt('--rerank_policy'), 'error_bound')
                self.assertEqual(opt('--rerank_float_refine'), '1')
                self.assertEqual(opt('--rerank_beam_width'), '8')
                self.assertNotIn('--inline_rerank', row)
                self.assertNotIn('--rerank_interval_max', row)
                self.assertEqual(opt('--filter_mode'), 'auto')
                self.assertEqual(opt('--entry_sampling'), 'navigation')
                self.assertEqual(opt('--navigation_file'), str(Path(opt('--graph_file')).parent / 'navigation.bin'))
                self.assertEqual(opt('--navigation_ef'), '64')
                self.assertEqual(opt('--navigation_budget'), '256')
                self.assertEqual(opt('--navigation_count'), '16')
                self.assertEqual(opt('--rabitq_bits'), '1')
                self.assertIn('centroids'+nc+'.fvecs', opt('--rabitq_centroids'))
                self.assertIn('--per_query_cache', row)
                self.assertNotIn('--rerank_error_bound', row)
                self.assertNotIn('--rerank_bound_alpha', row)

    def test_bits_based_rerank_is_independent_of_ef(self):
        for ds in ['arxiv', 'sift1m']:
            for bits, expected in [('1', '400'), ('2', '100'), ('3', '50')]:
                rows = self.search(ds, '--dry_run',
                                 '--rerank_policy', 'fixed', '--rabitq_bits', bits, '--efs', '30,100,1000')
                self.assertTrue(rows)
                for row in rows:
                    self.assertEqual(row[row.index('--rerank_num')+1], expected)
            rows = self.search(ds, '--dry_run',
                             '--rerank_policy', 'fixed', '--rabitq_bits', '3', '--k', '10', '--efs', '30')
            for row in rows:
                self.assertEqual(row[row.index('--rerank_num')+1], '20')

    def test_navigation_overrides_and_explicit_sampling(self):
        row = self.search('arxiv', '--dry_run', '--efs', '10',
                          '--navigation_file', '/tmp/new graph/navigation.bin',
                          '--navigation_ef', '128', '--navigation_budget', '64',
                          '--navigation_count', '8')[0]
        for key, value in [('navigation_file', '/tmp/new graph/navigation.bin'),
                           ('navigation_ef', '128'), ('navigation_budget', '64'), ('navigation_count', '8')]:
            self.assertEqual(row[row.index('--' + key) + 1], value)
        for sampling in ('guided', 'global', 'range'):
            row = self.search('arxiv', '--dry_run', '--efs', '10', '--entry_sampling', sampling)[0]
            self.assertEqual(row[row.index('--entry_sampling') + 1], sampling)
            self.assertNotIn('--navigation_file', row)
        for mode in ('prefilter', 'legacy'):
            row = self.search('arxiv', '--dry_run', '--efs', '10', '--filter_mode', mode)[0]
            self.assertNotIn('--navigation_file', row)
        for args in [('--navigation_count', '65'), ('--navigation_ef', '65537'),
                     ('--navigation_budget', '15'), ('--entry_sampling', 'unknown')]:
            self.search('arxiv', '--dry_run', *args, exit=2)

    def test_missing_navigation_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph, meta = root / 'graph.bin', root / 'meta.bin'
            graph.touch(); meta.touch()
            output = root / 'uncreated-output'
            result = subprocess.run(['bash', str(ROOT / 'scripts/disk_search_common.sh'), 'arxiv',
                '--graph_file', str(graph), '--meta_file', str(meta), '--out_dir', str(output)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn('navigation.bin', result.stderr)
            self.assertFalse(output.exists())

    def test_overrides_and_spaces(self):
        row = self.search('sift1m', '--dry_run',
                        '--data_root', '/tmp/data with spaces', '--model_dir', '/tmp/model space',
                        '--selectivities', '4', '--efs', '30', '--rerank_num', '50',
                        '--rabitq_bits', '3', '--filter_mode', 'graph', '--rerank_error_bound',
                        '--cross_query_cache', '--ml_graph_file', '/tmp/layout space.bin')[0]
        self.assertIn('/tmp/model space/sift_wp1_graph.bin', row)
        self.assertIn('/tmp/data with spaces/sift_query_range_2-4.ibin', row)
        self.assertIn('/tmp/layout space.bin.layout', row)
        self.assertIn('--rerank_error_bound', row)
        self.assertNotIn('--per_query_cache', row)
        self.assertEqual(row[row.index('--rerank_num')+1], '50')

    def test_invalid_before_io(self):
        for helper in ('disk_build_common.sh', 'disk_search_common.sh'):
            self.call(helper, exit=2)
            self.call(helper, 'unknown', '--dry_run', exit=2)
        for args in [('--efs',), ('--efs','0'), ('--rabitq_bits','5'),
                     ('--rerank_sq_refine','1'), ('--rerank_num','24'), ('--rerank_float_refine','2'),
                     ('--query_rng','x'), ('--rerank_batch_blocks','0'),
                     ('--prefilter_rerank','1'), ('--rerank_bound_alpha','nan'),
                     ('--rerank_bound_alpha','1'), ('--rerank_bound_alpha','2'),
                     ('--selectivities','2')]:
            self.search('sift1m', '--dry_run', *args, exit=2)

    def test_rerank_factor_forwarding(self):
        for ds in ['arxiv', 'sift1m']:
            for k in ['10','25']:
                for factor in ['1','3']:
                    rows=self.search(ds,'--dry_run',
                        '--k',k,'--rerank_factor',factor,'--efs','10,30,320',
                        '--inputs_dir','/tmp/factor-inputs','--selectivities','p1')
                    self.assertEqual(len(rows),3)
                    for row in rows:
                        self.assertEqual(row[row.index('--rerank_num')+1],'0')
                        self.assertEqual(row[row.index('--rerank_factor')+1],factor)
                        self.assertEqual(row[row.index('--k')+1],k)
            rows=self.search(ds,'--dry_run','--rerank_num','25')
            self.assertTrue(rows)
            for args in [('--rerank_factor','0'),('--rerank_factor','-1'),('--rerank_factor','1.5'),
                         ('--rerank_factor','3','--rerank_num','30')]:
                self.search(ds,'--dry_run',*args,exit=2)

    def test_dynamic_rerank_forwarding(self):
        for ds in ['arxiv','sift1m']:
            for policy in ['step','linear']:
                rows=self.search(ds,'--dry_run',
                    '--rerank_policy',policy,'--k','10','--efs','10,30,320',
                    '--inputs_dir','/tmp/dynamic-inputs','--selectivities','p1')
                self.assertEqual(len(rows),3)
                for row in rows:
                    self.assertEqual(row[row.index('--rerank_num')+1],'0')
                    self.assertEqual(row[row.index('--rerank_policy')+1],policy)
                    self.assertNotIn('--rerank_factor',row)
                for args in [('--rerank_factor','2'),('--rerank_num','40')]:
                    self.search(ds,'--dry_run',
                              '--rerank_policy',policy,*args,exit=2)

    def test_retired_inline_controls_do_not_reenable_or_reach_binary(self):
        dataset = 'arxiv'
        for args in [('--inline_rerank', 'off'), ('--no_inline_rerank',)]:
            rows = self.search(dataset, '--dry_run', *args)
            self.assertTrue(rows)
            self.assertTrue(all('--inline_rerank' not in row for row in rows))
        for args in [('--inline_rerank', 'on'), ('--inline_rerank', 'auto'),
                     ('--inline_rerank_efs_factor', '2'), ('--rerank_interval_max', '8')]:
            self.search(dataset, '--dry_run', *args, exit=2)

    def test_build_paths(self):
        for ds, prefix in [('arxiv','wp1'), ('sift1m','sift_wp1')]:
            row = self.build(ds, '--dry_run', '--out_dir', '/tmp/out dir')[0]
            self.assertIn('/tmp/out dir/'+prefix+'_graph.bin', row)
            for key, val in [('--m','32'),('--efc','256'),('--wp','5'),('--o','4'),
                             ('--build_mem_gb','1.0'),('--win_anchor','top'),
                             ('--resident_builder','left_stream'),('--resident_batch','256')]:
                self.assertEqual(row[row.index(key)+1],val)
        self.build('arxiv','--dry_run','--threads', exit=2)

    def test_retired_build_modes_fail_before_io(self):
        with tempfile.TemporaryDirectory() as d:
            for args in [('--resident_builder', 'two_pass'), ('--resident_builder', 'unknown'),
                         ('--copy_top',), ('--wow_online_compat',),
                         ('--wow_passes', '0'), ('--wow_passes', '2'), ('--resident_batch', '0')]:
                self.build('arxiv', '--dry_run', '--out_dir', d, *args, exit=2)
                self.assertFalse(list(Path(d).iterdir()))
            row = self.build('arxiv', '--dry_run', '--wow_passes', '1')[0]
            self.assertNotIn('--wow_passes', row)
            self.assertEqual(row[row.index('--resident_builder') + 1], 'left_stream')

    def test_skip_missing_layout_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'graph').touch(); (p/'meta').touch()
            self.search('sift1m', '--graph_file',str(p/'graph'),
                      '--meta_file',str(p/'meta'), '--skip_search','--skip_shuffle',
                      '--out_dir',d,exit=2)

def write_receipt(graph, layout, meta):
    value = dict(status='complete', protocol_version=1, ml_only=True, files=[
        dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in (graph, layout, meta)])
    Path(str(graph) + '.completion.json').write_text(json.dumps(value))
    return value


class ShuffleReceipt(unittest.TestCase):
    def test_committed_files_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name for name in ('graph', 'layout', 'meta')]
            for i, path in enumerate(paths): path.write_bytes(bytes([i]) * 16)
            record = write_receipt(*paths)
            self.assertEqual(receipt.validate_completion(*paths), record)
            path = Path(str(paths[0]) + '.completion.json')
            for alter in [lambda r: r.update(status='writing'),
                          lambda r: r.update(files=r['files'][:2]),
                          lambda r: r['files'][1].update(path=r['files'][0]['path']),
                          lambda r: r['files'][0].update(bytes=15),
                          lambda r: r['files'][0].update(sha256='0' * 64),
                          lambda r: r['files'][0].update(path='relative.graph')]:
                changed = json.loads(json.dumps(record)); alter(changed)
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): receipt.validate_completion(*paths)
            path.unlink()
            with self.assertRaises(OSError): receipt.validate_completion(*paths)


if __name__ == '__main__':
    unittest.main()
