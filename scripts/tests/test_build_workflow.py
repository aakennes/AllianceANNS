"""Lifecycle contracts: failed builds keep resume open; successful ones verify first."""
import argparse
from contextlib import redirect_stderr
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchlib.build_artifacts import DiskObservation, disk_snapshot, finalization_argv
from benchlib.shuffle_memory import scope_memory_properties, ShuffleRssObservation, rss_measurements
spec = importlib.util.spec_from_file_location('dataset_runner', ROOT / 'run_dataset.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BuildWorkflow(unittest.TestCase):
    def test_shuffle_unlimited_scope_and_rss_resets(self):
        self.assertEqual(scope_memory_properties(None), ['-p', 'MemoryMax=infinity'])
        self.assertIn('MemorySwapMax=0', scope_memory_properties(600 << 20))
        for bad in (0, -1, 'null', True):
            with self.assertRaises(ValueError):
                scope_memory_properties(bad)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, name, children in ((10, 'systemd-run', '11'), (11, 'alliance_shuffl', '')):
                task = root / str(pid)
                (task / 'task' / str(pid)).mkdir(parents=True)
                (task / 'comm').write_text(name)
                (task / 'task' / str(pid) / 'children').write_text(children)
            status = root / '11/status'
            status.write_text('VmRSS:\t2048 kB\nVmHWM:\t4096 kB\n')
            observer = ShuffleRssObservation(10, root)
            observer.sample()
            status.write_text('VmRSS:\t1024 kB\nVmHWM:\t1024 kB\n')
            observer.sample()
            self.assertEqual(observer.peak, 4 << 20)
            self.assertEqual(rss_measurements('[RSS] shuffle-orders peak 5.0 MB', 1, observer)['peak_rss_mib'], 5)

    def test_retired_pipe_options_fail_before_loading_or_execution(self):
        for options in (['--algorithm', 'pipeannfilter'],
                        ['--entry-algorithm', 'pipeannfilter'],
                        ['--cache-bytes', '4096']):
            with self.subTest(options=options), \
                    patch.object(sys, 'argv', ['run_dataset.py', 'search', 'arxiv2m',
                                             '--dry-run'] + options), \
                    patch.object(runner, 'load_config') as load, \
                    patch.object(runner, 'run_command') as run, redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as failed:
                    runner.main()
                self.assertEqual(failed.exception.code, 2)
                self.assertIn(options[0], error.getvalue())
                load.assert_not_called()
                run.assert_not_called()

    def test_alliance_wrapper_algorithm_remains_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'new_build'
            config = dict(dataset='arxiv2m', paths={'index_run_root': directory},
                build={'allianceann': dict(binary='build', parameters=dict(
                    basevec='base.bin', baseatt='attrs.bin', threads=32),
                    memory_max_bytes=1 << 30, upper_builder='wow')})
            with patch.object(sys, 'argv', ['run_dataset.py', 'build', 'arxiv2m',
                    '--entry-algorithm', 'allianceann', '--algorithm', 'allianceann',
                    '--dry-run', '--out-dir', str(out)]), \
                    patch.object(runner, 'load_config', return_value=config), \
                    patch.object(runner, 'check_binary'), patch.object(runner, 'existing'), \
                    patch.object(runner, 'run_command', return_value=('', 0)) as run:
                runner.main()
            self.assertEqual(run.call_args[0][1], 'build_alliance')
            self.assertFalse(out.exists())

    def test_search_keeps_native_and_frozen_commands_and_metric_wise_medians(self):
        for driver in ('main', 'frozen'):
            with self.subTest(driver=driver), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                out = root / 'run'
                out.mkdir()
                params = dict(graph_file='graph', meta_file='meta', ml_graph_file='ml',
                              rabitq_centroids='centroids', rabitq_cids='cids', entry_sampling='guided')
                section = dict(binary='search', driver=driver, parameters=params,
                               quant_checkpoint='checkpoint', memory_max_bytes=1 << 30)
                query = dict(query_vec='queries', query_rng='ranges', gt={'10': 'gt'}, layout='cache')
                settings = dict(allianceann=section, queries={'p4': {'allianceann': query}},
                                threads=16, ef_profiles={}, fallback_efs={'10': [10]},
                                recall_stop=.995, rerank_policy='error_bound')
                config = dict(dataset='arxiv2m', paths={'result_root': str(root / 'results')},
                              search=settings)
                args = argparse.Namespace(algorithm='allianceann', workload='p4', k=10,
                    efs=[10], index_dir=None, repeats=3, dry_run=False, config=Path('config'))
                context = runner.Context((args, config, out))
                measured = [dict(efs=10, unique_recall=.9, qps=qps, total_io=graph+1,
                                 io_submits=graph, rerank_io_submits=1, short=0)
                            for qps, graph in ((1, 9), (3, 3), (5, 12))]
                with patch.object(runner, 'check_binary'), patch.object(runner, 'existing'), \
                        patch.object(runner, 'run_command', return_value=('log', 12)) as run, \
                        patch.object(runner, 'parse_alliance', side_effect=[[r] for r in measured]):
                    runner.search(context)
                self.assertEqual(run.call_count, 3)
                command, name = run.call_args_list[0][0][:2]
                self.assertEqual(command[:4], ['taskset', '-c', '1-16', 'search'])
                self.assertEqual(name, 'search_r1')
                environment = run.call_args_list[0][0][4]
                if driver == 'main':
                    self.assertNotIn('--efs', command)
                    self.assertEqual(command[command.index('--efs_list') + 1], '10')
                    self.assertEqual(command[command.index('--quant_checkpoint') + 1], 'checkpoint')
                    self.assertEqual(environment, {})
                else:
                    self.assertEqual(command[command.index('--efs') + 1], '10')
                    self.assertNotIn('--quant_checkpoint', command)
                    self.assertEqual(environment, dict(FINAL_QUANT_CHECKPOINT='checkpoint',
                        FINAL_EFS='10', FINAL_RECALL_STOP='0.995'))
                csv_path = next((root / 'results').rglob('*.csv'))
                with csv_path.open(encoding='utf-8-sig', newline='') as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual(len(rows), 1)
                self.assertEqual(float(rows[0]['qps']), 3)
                self.assertEqual(float(rows[0]['io_times']), 10)
                self.assertEqual(float(rows[0]['graph_io']), 9)
                self.assertEqual(int(rows[0]['repeats']), 3)

    def test_finalize_only_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = argparse.Namespace(algorithm='allianceann', dry_run=False)
            parameters = dict(basevec='base.bin', baseatt='attrs.bin', basevec_format='bin',
                              rawvec_export='off', space='ip')
            config = dict(dataset='synthetic', build={'allianceann': dict(binary='build',
                parameters=parameters, memory_max_bytes=1 << 30, upper_builder='wow',
                finalize_on_success=True, verifier_binary='verify')})
            context = runner.Context((args, config, out))
            calls = []
            def run(argv, name, *_):
                calls.append((name, argv))
                return '', 0
            with patch.object(runner, 'check_binary'), patch.object(runner, 'existing'), \
                    patch.object(runner, 'run_command', side_effect=run):
                runner.build(context)
            self.assertEqual([c[0] for c in calls], ['build_alliance', 'finalize_build'])
            final = calls[-1][1]
            self.assertEqual(final[final.index('--format') + 1], 'bin')
            self.assertIn('--close-resume-window', final)
            manifest = json.loads((out / 'index.json').read_text())
            self.assertTrue(manifest['resume_closed'])
            self.assertEqual(manifest['rawvec_export'], 'off')
            (out / 'index.json').unlink()
            with patch.object(runner, 'check_binary'), patch.object(runner, 'existing'), \
                    patch.object(runner, 'run_command', side_effect=RuntimeError('build failure')) as run:
                with self.assertRaisesRegex(RuntimeError, 'build failure'):
                    runner.build(context)
                self.assertEqual(run.call_count, 1)
            self.assertFalse((out / 'index.json').exists())

    def test_finalizer_failure_does_not_publish_index(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = argparse.Namespace(algorithm='allianceann', dry_run=False)
            section = dict(binary='build', parameters=dict(basevec='base.bin',
                baseatt='attrs.bin', rawvec_export='off'), memory_max_bytes=1 << 30,
                upper_builder='wow', finalize_on_success=True, verifier_binary='verify')
            context = runner.Context((args, dict(dataset='synthetic',
                                               build={'allianceann': section}), out))
            artifacts = [out / 'graph.bin.buildvecs', out / 'checkpoint.bin']
            for path in artifacts:
                path.write_bytes(b'resume evidence')
            with patch.object(runner, 'check_binary'), patch.object(runner, 'existing'), \
                    patch.object(runner, 'run_command', side_effect=[('', 0),
                        RuntimeError('verification refused')]) as run:
                with self.assertRaisesRegex(RuntimeError, 'verification refused'):
                    runner.build(context)
            self.assertEqual([call[0][1] for call in run.call_args_list],
                             ['build_alliance', 'finalize_build'])
            self.assertFalse((out / 'index.json').exists())
            for path in artifacts:
                self.assertEqual(path.read_bytes(), b'resume evidence')

    def test_shuffle_manifest_is_independent_and_cache_follows_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = argparse.Namespace(algorithm='allianceann', dry_run=False,
                                      index_dir=None, workload='origin')
            config = dict(dataset='synthetic', index=dict(graph_file='source/graph.bin',
                meta_file='source/meta.bin'), shuffle=dict(binary='shuffle',
                parameters=dict(space='l2'), memory_max_bytes=1 << 30,
                collect_binary='collect', cache_binary='prepare', static_budget_bytes=4096),
                search=dict(queries={'origin': {'allianceann': {'query_rng': 'ranges.bin'}}}))
            context = runner.Context((args, config, out))
            events = []

            def command(argv, name, *_):
                self.assertFalse((out / 'index.json').exists())
                events.append(name)
                return '', 0

            def verified(*paths):
                events.append('verified_receipt')

            with patch.object(runner, 'check_binary'), patch.object(runner, 'existing'), \
                    patch.object(runner, 'run_command', side_effect=command) as run, \
                    patch.object(runner, 'validate_completion', side_effect=verified) as validate:
                runner.shuffle(context)
            self.assertEqual(events, ['shuffle', 'verified_receipt', 'collect_landing', 'static_cache'])
            validate.assert_called_once_with(out / 'ml_graph.bin', out / 'ml_graph.bin.layout',
                                             out / 'ml_meta.bin')
            self.assertEqual([call[0][0][0] for call in run.call_args_list],
                             ['shuffle', 'collect', 'prepare'])
            self.assertNotIn('--static_cache_budget', run.call_args_list[0][0][0])
            manifest = json.loads((out / 'index.json').read_text())
            self.assertTrue(manifest['ml_only'])
            self.assertEqual(manifest['meta_file'], str(out / 'ml_meta.bin'))
            self.assertEqual(manifest['ml_graph_file'], str(out / 'ml_graph.bin'))
            self.assertEqual(manifest['static_layout'], str(out / 'static_4096.layout'))
            self.assertEqual(manifest['shuffle_completion'], str(out / 'ml_graph.bin.completion.json'))

            # An uncommitted layout must not prepare a cache or advertise an index.
            (out / 'index.json').unlink()
            with patch.object(runner, 'check_binary'), patch.object(runner, 'existing'), \
                    patch.object(runner, 'run_command', return_value=('', 0)) as run, \
                    patch.object(runner, 'validate_completion', side_effect=ValueError('incomplete')):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    runner.shuffle(context)
            self.assertEqual(run.call_count, 1)
            self.assertFalse((out / 'index.json').exists())

    def test_sparse_files_and_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'sparse'
            with data.open('wb') as output:
                output.truncate(8 << 20)
            os.link(str(data), str(root / 'alias'))
            (root / 'outside').symlink_to('/dev/zero')
            snapshot = disk_snapshot(root)
            self.assertEqual(snapshot['logical_bytes'], 8 << 20)
            self.assertEqual(len(snapshot['files']), 1)
            self.assertLess(snapshot['allocated_bytes'], snapshot['logical_bytes'])
            observer = DiskObservation(root, 'build_fixture')
            observer.sample(0)
            data.unlink()
            (root / 'alias').unlink()
            observer.sample(1)
            record = observer.close()
            self.assertEqual(record['sample_count'], 2)
            self.assertGreaterEqual(record['observed_peak_logical']['logical_bytes'], 8 << 20)
            rows = [json.loads(line) for line in observer.path.read_text().splitlines()]
            self.assertTrue(all('files' not in row for row in rows))

    def test_automatic_finalizer_requires_real_attributes(self):
        with self.assertRaises(ValueError):
            finalization_argv('python', 'finalizer', 'verifier', 'out', {'baseatt': 'serial'})


if __name__ == '__main__':
    unittest.main()
