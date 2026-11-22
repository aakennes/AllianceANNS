"""Identity checks and paired exports use tiny files, never ANN production data."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchlib import runner
from benchlib.export import finalize_experiment
from benchlib.spec import digest_file, file_stat, fingerprint_inputs, validate_spec


class SpecExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR"))
        self.root = Path(self.tmp.name)
        binary = self.root / "driver"
        binary.write_bytes(b"synthetic driver identity")
        self.asset = self.root / "fixture"
        self.asset.write_bytes(b"small fixture data")
        self.spec = {
            "schema_version": 2, "id": "synthetic", "memory": {"loading_limit_bytes": 200, "search_rss_limit_bytes": 100},
            "aggregation": {"policy": "qps-paired-trim", "repeats": 5},
            "lock_paths": [str(self.root / "global.lock")],
            "inputs": [{"path": str(binary), "sha256": digest_file(binary)},
                       {"path": str(self.asset), "sha256": digest_file(self.asset)}],
            "curves": [{"id": "origin_allianceann_landing", "algorithm": "allianceann", "workload": "origin",
                        "cache_policy": "landing", "efs": [10], "repeats": 5, "queries": 2, "bits": 1, "k": 1,
                        "argv": [str(binary)], "env": {"CASELAW_OUTPUT": "{output}"}, "scope_prefix": "caselaw-rss4-r12-",
                        "inputs": {key: str(self.asset) for key in ("base", "query", "ranges", "gt", "attributes")}, "expected": {}}]}

    def tearDown(self):
        self.tmp.cleanup()

    def native_spec(self):
        spec = deepcopy(self.spec)
        spec.update(schema_version=3, native_protocol="alliance-formal-v1", adapter="dataset-native-v1")
        spec["memory"]["mode"] = "whole_process"
        spec["curves"][0].update(native_protocol="alliance-formal-v1", metric="l2",
                                 scope_prefix="alliance-bench-", quantization_fingerprint="123",
                                 input_formats={"base": "fvecs", "attributes": "raw-i32"})
        spec["curves"][0]["env"]["BENCH_EXPECTED_FP"] = "123"
        return spec

    def rows(self):
        rows = []
        for repeat, qps in enumerate([10, 50, 30, 20, 40], 1):
            rows.append(dict(curve_id="origin_allianceann_landing", algorithm="allianceann", workload="origin",
                             cache_policy="landing", bits=1, ef=10, repeat=repeat, queries=2, validated=True,
                             qps=qps, recall=0.9, p95_ms=1.0, io_requests_per_query=repeat * 10,
                             io_4k_equivalent_per_query=repeat * 20, io_8k_equivalent_per_query=repeat * 10,
                             io_bytes_per_query=repeat * 20 * 4096, batch_wall_seconds=2.0 / qps,
                             search_peak_rss_bytes=99, loading_peak_rss_bytes=150))
        return rows

    def test_identity_and_new_capture_are_distinct(self):
        expected = fingerprint_inputs(self.spec)
        self.assertEqual(expected[str(self.asset)]["expected_sha256"], digest_file(self.asset))
        self.spec["inputs"][1].update(sha256=None, role="index", identity_mode="stat_then_sha256",
                                      expected_stat=file_stat(self.asset), provenance={"request_sha256": "a" * 64})
        captured = fingerprint_inputs(self.spec)[str(self.asset)]
        self.assertEqual(captured["identity_mode"], "stat_then_sha256")
        self.assertIsNone(captured["expected_sha256"])
        self.asset.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "stat changed"):
            fingerprint_inputs(self.spec)

    def test_unbound_binary_and_invalid_paths_rejected(self):
        bad = deepcopy(self.spec)
        bad["inputs"][0]["sha256"] = None
        with self.assertRaises(ValueError):
            validate_spec(bad)
        bad = deepcopy(self.spec)
        bad["curves"][0]["id"] = "../escape"
        with self.assertRaises(ValueError):
            validate_spec(bad)
        bad = deepcopy(self.spec)
        bad["curves"][0]["argv"].append("{unknown}")
        with self.assertRaises(ValueError):
            validate_spec(bad)
        bad["curves"][0]["argv"][-1] = "/absolute/{unknown}"
        with self.assertRaises(ValueError):
            validate_spec(bad)

    def test_retired_options_rejected_only_for_in_tree_execution(self):
        native = self.native_spec()
        native["curves"][0]["argv"] += ["--memory_budget_bytes", "0", "--entry_sampling", "guided",
                                        "--entry_samples", "256", "--entry_count", "4"]
        self.assertIs(validate_spec(native), native)
        for flag in ("--memory", "--enum_thresh", "--export_threads", "--exact", "--basevec"):
            for arguments in ([flag], [flag, "1"], [flag + "=1"]):
                with self.subTest(arguments=arguments):
                    bad = deepcopy(native)
                    bad["curves"][0]["argv"] += arguments
                    with self.assertRaisesRegex(ValueError, "retired in-tree search options: " + flag):
                        validate_spec(bad)
                    historical = deepcopy(self.spec)
                    historical["curves"][0]["argv"] += arguments
                    self.assertIs(validate_spec(historical), historical)

    def test_retired_plan_stops_before_input_io_job_creation_and_launch(self):
        bad = self.native_spec()
        bad["curves"][0]["argv"] += ["--memory", "1"]
        job = self.root / "retired-job"
        with mock.patch.object(Path, "is_file", side_effect=AssertionError("input stat")) as stat, \
                mock.patch.object(Path, "open", side_effect=AssertionError("input open")) as opened, \
                mock.patch.object(runner, "_mkdir_durable", side_effect=AssertionError("job creation")) as mkdir, \
                mock.patch.object(runner, "_fingerprints", side_effect=AssertionError("fingerprinting")) as fingerprints, \
                mock.patch.object(runner, "execute_curve", side_effect=AssertionError("native launch")) as execute:
            with self.assertRaisesRegex(ValueError, "retired in-tree search options"):
                fingerprint_inputs(bad)
            for resume in (False, True):
                with self.subTest(resume=resume), self.assertRaisesRegex(ValueError, "retired in-tree search options"):
                    runner.run_experiment(bad, job, resume=resume)
            for boundary in (stat, opened, mkdir, fingerprints, execute):
                boundary.assert_not_called()
        self.assertFalse(job.exists())

    def test_navigation_requires_fixed_identity_and_valid_budgets(self):
        native = self.native_spec()
        native['curves'][0]['argv'] += ['--navigation_file', str(self.asset), '--navigation_ef', '64',
                                       '--navigation_budget', '256', '--navigation_count', '16']
        self.assertIs(validate_spec(native), native)
        independent_budget = deepcopy(native)
        argv = independent_budget['curves'][0]['argv']
        argv[argv.index('--navigation_ef')+1] = '128'
        argv[argv.index('--navigation_budget')+1] = '64'
        self.assertIs(validate_spec(independent_budget), independent_budget)
        unbound = deepcopy(native)
        unbound['curves'][0]['argv'][2] = str(self.root / 'unbound-navigation')
        with self.assertRaisesRegex(ValueError, 'pinned SHA256'):
            validate_spec(unbound)
        captured = deepcopy(native)
        captured['inputs'][1].update(sha256=None, role='index', identity_mode='stat_then_sha256',
                                     expected_stat=file_stat(self.asset), provenance={'source':'fixture'})
        with self.assertRaisesRegex(ValueError, 'pinned SHA256'):
            validate_spec(captured)
        for flag,value in [('--navigation_ef','0'),('--navigation_ef','-1'),('--navigation_ef','1.5'),
                           ('--navigation_count','65'),('--navigation_budget','15'),
                           ('--navigation_ef','65537'),('--navigation_budget','65537')]:
            bad = deepcopy(native)
            argv = bad['curves'][0]['argv']
            argv[argv.index(flag)+1] = value
            with self.subTest(flag=flag,value=value), self.assertRaises(ValueError):
                validate_spec(bad)
        for suffix in (['--navigation_file'],['--navigation_file',str(self.asset)],['--navigation_ef=64']):
            bad = deepcopy(native)
            bad['curves'][0]['argv'] += suffix
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                validate_spec(bad)

    def test_default_and_explicit_navigation_require_bound_paths(self):
        ml = ['--pipeline', '1', '--ml_graph_file', str(self.asset), '--ml_meta_file', str(self.asset)]
        cases = [['--entry_sampling', 'navigation']]
        for mode in ('graph', 'auto'):
            cases.append(ml + ['--filter_mode', mode])
            cases.append(ml + ['--filter_mode', mode, '--entry_sampling', 'navigation'])
            cases.append(ml + ['--ml_only', '--filter_mode', mode])
        for arguments in cases:
            with self.subTest(arguments=arguments):
                native = self.native_spec()
                native['curves'][0]['argv'] += arguments
                with self.assertRaisesRegex(ValueError, 'explicit --navigation_file.*pinned SHA256'):
                    validate_spec(native)
                native['curves'][0]['argv'] += ['--navigation_file', str(self.root / 'unbound-navigation')]
                with self.assertRaisesRegex(ValueError, 'pinned SHA256'):
                    validate_spec(native)
                native['curves'][0]['argv'][-1] = str(self.asset)
                self.assertIs(validate_spec(native), native)
                historical = deepcopy(self.spec)
                historical['curves'][0]['argv'] += arguments
                self.assertIs(validate_spec(historical), historical)

    def test_non_navigation_routes_keep_their_input_contract(self):
        ml = ['--pipeline', '1', '--ml_graph_file', str(self.asset), '--ml_meta_file', str(self.asset)]
        cases = [[], ['--pipeline', '1'], ml + ['--filter_mode', 'prefilter'],
                 ml + ['--filter_mode', 'legacy'], ml + ['--pipeline', '0'],
                 ml + ['--pipeline', '00', '--filter_mode', 'graph']]
        for route in ('graph', 'auto'):
            for sampling in ('guided', 'range', 'global'):
                cases.append(ml + ['--filter_mode', route, '--entry_sampling', sampling])
        cases.append(ml + ['--filter_mode', 'graph', '--entry_sampling', 'navigation',
                          '--entry_sampling', 'guided'])
        cases.append(ml + ['--filter_mode', 'graph', '--filter_mode', 'prefilter'])
        for arguments in cases:
            with self.subTest(arguments=arguments):
                native = self.native_spec()
                native['curves'][0]['argv'] += arguments
                self.assertIs(validate_spec(native), native)

    def test_implicit_navigation_stops_before_fingerprinting_or_launch(self):
        bad = self.native_spec()
        bad['curves'][0]['argv'] += ['--pipeline', '1', '--filter_mode', 'auto',
                                    '--ml_graph_file', str(self.asset), '--ml_meta_file', str(self.asset)]
        job = self.root / 'unbound-navigation-job'
        with mock.patch.object(Path, 'is_file', side_effect=AssertionError('input stat')) as stat, \
                mock.patch.object(Path, 'open', side_effect=AssertionError('input open')) as opened, \
                mock.patch.object(runner, '_mkdir_durable', side_effect=AssertionError('job creation')) as mkdir, \
                mock.patch.object(runner, '_fingerprints', side_effect=AssertionError('fingerprinting')) as fingerprints, \
                mock.patch.object(runner, 'execute_curve', side_effect=AssertionError('native launch')) as execute:
            with self.assertRaisesRegex(ValueError, 'pinned SHA256'):
                fingerprint_inputs(bad)
            for resume in (False, True):
                with self.subTest(resume=resume), self.assertRaisesRegex(ValueError, 'pinned SHA256'):
                    runner.run_experiment(bad, job, resume=resume)
            for boundary in (stat, opened, mkdir, fingerprints, execute):
                boundary.assert_not_called()
        self.assertFalse(job.exists())

    def test_all_five_validate_and_same_three_are_exported(self):
        job = self.root / "job"
        job.mkdir()
        result = finalize_experiment(self.spec, job, self.rows())
        out = job / "aggregate_export"
        decision = json.loads((out / "trim_decisions.json").read_text())[0]
        self.assertEqual(decision["kept_repeat_ids"], [4, 3, 5])
        self.assertEqual(decision["metrics"]["qps"], 30)
        self.assertEqual(decision["metrics"]["io_requests_per_query"], 40)
        self.assertEqual(result["queries"], 10)
        manifest = json.loads((out / "export_manifest.json").read_text())
        self.assertTrue(all(digest_file(out / name) == digest for name, digest in manifest.items()))
        self.assertEqual(finalize_experiment(self.spec, job, self.rows()), result)
        (out / "export_manifest.json.tmp").write_text("interrupted manifest write")
        self.assertEqual(finalize_experiment(self.spec, job, self.rows()), result)
        bad = self.rows()
        bad[0]["validated"] = False
        with self.assertRaisesRegex(ValueError, "unvalidated"):
            finalize_experiment(self.spec, job, bad)

    def test_missing_repeat_memory_failure_and_identity_mismatch(self):
        for mutate in (lambda rows: rows.pop(), lambda rows: rows[0].update(search_peak_rss_bytes=101),
                       lambda rows: rows[0].update(queries=3)):
            rows = self.rows()
            mutate(rows)
            with self.assertRaises(ValueError):
                finalize_experiment(self.spec, self.root, rows)

    @unittest.skipIf(os.name == "nt", "unprivileged Windows symlink creation is not portable")
    def test_export_directory_symlink_rejected(self):
        job = self.root / "job"
        job.mkdir()
        external = self.root / "external"
        external.mkdir()
        (job / "aggregate_export").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside its job"):
            finalize_experiment(self.spec, job, self.rows())
        self.assertEqual(list(external.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
