"""Navigation binding and preflight; no native executable or real queries."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_dataset as runner


class NavigationWorkflow(unittest.TestCase):
    def fixture(self, root, sampling='navigation'):
        paths = {}
        for name in ('graph_file', 'meta_file', 'ml_graph_file', 'rabitq_centroids',
                     'rabitq_cids', 'query_vec', 'query_rng', 'gt', 'layout', 'checkpoint'):
            path = root / name
            path.touch()
            paths[name] = str(path)
        params = {key: paths[key] for key in ('graph_file', 'meta_file', 'ml_graph_file',
                                             'rabitq_centroids', 'rabitq_cids')}
        params.update(pipeline=1, filter_mode='auto')
        if sampling is not None:
            params['entry_sampling'] = sampling
        section = dict(binary='search', driver='main', parameters=params,
                       quant_checkpoint=paths['checkpoint'], memory_max_bytes=1 << 30)
        query = dict(query_vec=paths['query_vec'], query_rng=paths['query_rng'],
                     gt={'10': paths['gt']}, layout=paths['layout'])
        settings = dict(allianceann=section, queries={'origin': {'allianceann': query}},
                        threads=16, ef_profiles={}, fallback_efs={'10': [10]}, recall_stop=.995)
        args = argparse.Namespace(algorithm='allianceann', workload='origin', k=10, efs=[10],
                                  index_dir=None, repeats=1, dry_run=True)
        return runner.Context((args, dict(dataset='arxiv2m', search=settings), root / 'new_output'))

    def command(self, context):
        with patch.object(runner, 'check_binary'), \
                patch.object(runner, 'run_command', return_value=('', 0)) as run:
            runner.search(context)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(context[2].exists())
        return run.call_args[0][0]

    def manifest(self, root, context, **overrides):
        params = context[1]['search']['allianceann']['parameters']
        value = dict(algorithm='allianceann', dataset='arxiv2m',
                     graph_file=params['graph_file'], meta_file=params['meta_file'],
                     ml_graph_file=params['ml_graph_file'], static_layout=str(root / 'layout'),
                     cache_workload='origin', ml_only=True)
        value.update(overrides)
        manifest_dir = root / 'manifest'
        manifest_dir.mkdir()
        (manifest_dir / 'index.json').write_text(json.dumps(value))
        context[0].index_dir = manifest_dir
        return manifest_dir

    def test_default_and_explicit_navigation_use_current_graph_directory(self):
        for sampling in (None, 'navigation'):
            with self.subTest(sampling=sampling), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                context = self.fixture(root, sampling)
                context[1]['search']['allianceann']['entry_sampling_explicit'] = sampling is not None
                nav = root / 'navigation.bin'
                nav.touch()
                argv = self.command(context)
                self.assertEqual(argv[argv.index('--entry_sampling') + 1], 'navigation')
                self.assertEqual(argv[argv.index('--navigation_file') + 1], str(nav.resolve()))
                self.assertNotIn('--entry_sampling_explicit', argv)

    def test_missing_navigation_fails_before_any_query_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            context = self.fixture(Path(directory))
            for dry_run in (True, False):
                context[0].dry_run = dry_run
                with patch.object(runner, 'check_binary'), patch.object(runner, 'run_command') as run:
                    with self.assertRaisesRegex(ValueError, 'Missing navigation asset for the selected index'):
                        runner.search(context)
                    run.assert_not_called()
                self.assertFalse(context[2].exists())

    def test_explicit_sampling_and_non_graph_routes_need_no_navigation(self):
        for sampling in ('guided', 'global', 'range'):
            with self.subTest(sampling=sampling), tempfile.TemporaryDirectory() as directory:
                argv = self.command(self.fixture(Path(directory), sampling))
                self.assertEqual(argv[argv.index('--entry_sampling') + 1], sampling)
                self.assertNotIn('--navigation_file', argv)
        for override in ({'filter_mode': 'prefilter'}, {'filter_mode': 'legacy'}, {'pipeline': 0}):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                context = self.fixture(Path(directory))
                section = context[1]['search']['allianceann']
                section['entry_sampling_explicit'] = False
                section['parameters'].update(override)
                argv = self.command(context)
                self.assertNotIn('--navigation_file', argv)
                self.assertEqual(argv[argv.index('--entry_sampling') + 1], 'guided')

    def test_explicit_navigation_rejects_unsupported_routes_before_execution(self):
        for override in ({'filter_mode': 'prefilter'}, {'filter_mode': 'legacy'},
                         {'pipeline': 0}, {'ml_graph_file': None}):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                context = self.fixture(Path(directory))
                section = context[1]['search']['allianceann']
                section['entry_sampling_explicit'] = True
                section['parameters'].update(override)
                with patch.object(runner, 'check_binary'), patch.object(runner, 'run_command') as run:
                    with self.assertRaisesRegex(ValueError, 'Explicit entry_sampling=navigation requires'):
                        runner.search(context)
                    run.assert_not_called()
                self.assertFalse(context[2].exists())

    def test_load_config_records_stage_override_separately_from_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            defaults = root / 'defaults.json'
            runner.save(defaults, dict(paths={}, environment={}, search_protocol={},
                        search=dict(driver='main', parameters=dict(entry_sampling='navigation'))))
            runner.save(root / 'dataset.json', dict(dataset='arxiv2m', index={}, paths={},
                                                    memory_max_bytes=1 << 30))
            runner.save(root / 'queries.json', dict(workloads={'origin': {}}))
            runner.save(root / 'efs.json', {})
            for sampling in (None, 'navigation', 'guided'):
                with self.subTest(sampling=sampling):
                    override = dict(filter_mode='prefilter')
                    if sampling is not None:
                        override['entry_sampling'] = sampling
                    runner.save(root / 'search.json', dict(parameters=override))
                    config = runner.load_config(root, 'search', 'origin', defaults)
                    section = config['search']['allianceann']
                    self.assertEqual(section['entry_sampling_explicit'], sampling is not None)
                    self.assertNotIn('entry_sampling_explicit', section['parameters'])
                    params = dict(section['parameters'])
                    if sampling == 'navigation':
                        with self.assertRaisesRegex(ValueError, 'Explicit entry_sampling=navigation requires'):
                            runner.resolve_navigation(params, entry_sampling_explicit=section['entry_sampling_explicit'])
                    else:
                        runner.resolve_navigation(params, entry_sampling_explicit=section['entry_sampling_explicit'])
                        self.assertEqual(params['entry_sampling'], 'guided')

    def test_manifest_navigation_relative_path_and_explicit_guided(self):
        for sampling in ('navigation', 'guided'):
            with self.subTest(sampling=sampling), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                context = self.fixture(root, sampling)
                params = context[1]['search']['allianceann']['parameters']
                params['navigation_file'] = str(root / 'stale-navigation.bin')
                manifest_dir = self.manifest(root, context, navigation_file='bound navigation.bin')
                nav = manifest_dir / 'bound navigation.bin'
                nav.touch()
                argv = self.command(context)
                self.assertNotIn(params['navigation_file'], argv)
                if sampling == 'guided':
                    self.assertNotIn('--navigation_file', argv)
                else:
                    self.assertEqual(argv[argv.index('--navigation_file') + 1], str(nav.resolve()))

    def test_new_manifest_never_falls_back_to_configured_old_navigation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self.fixture(root)
            old_nav = root / 'navigation.bin'
            old_nav.touch()
            context[1]['search']['allianceann']['parameters']['navigation_file'] = str(old_nav)
            new_graph = root / 'new_index' / 'graph.bin'
            new_graph.parent.mkdir()
            new_graph.touch()
            self.manifest(root, context, graph_file=str(new_graph))
            with patch.object(runner, 'check_binary'), patch.object(runner, 'run_command') as run:
                with self.assertRaisesRegex(ValueError, 'new_index.*navigation.bin'):
                    runner.search(context)
                run.assert_not_called()
            new_nav = new_graph.parent / 'navigation.bin'
            new_nav.touch()
            argv = self.command(context)
            self.assertEqual(argv[argv.index('--navigation_file') + 1], str(new_nav.resolve()))

    def test_ml_only_without_raw_graph_uses_selected_ml_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self.fixture(root)
            ml_graph = root / 'new_ml' / 'ml_graph.bin'
            ml_graph.parent.mkdir()
            ml_graph.touch()
            nav = ml_graph.parent / 'navigation.bin'
            nav.touch()
            self.manifest(root, context, graph_file=None, ml_graph_file=str(ml_graph))
            argv = self.command(context)
            self.assertNotIn('--graph_file', argv)
            self.assertEqual(argv[argv.index('--navigation_file') + 1], str(nav.resolve()))

    def test_shared_defaults_only_change_search_entry_policy(self):
        defaults = runner.read(ROOT / 'defaults.json')
        self.assertEqual(defaults['search']['parameters']['entry_sampling'], 'navigation')
        self.assertEqual(len(defaults['search']['sha256']), 64)
        self.assertNotIn('entry_sampling', defaults['build']['parameters'])
        self.assertNotIn('navigation_file', defaults['build']['parameters'])


if __name__ == '__main__':
    unittest.main()
