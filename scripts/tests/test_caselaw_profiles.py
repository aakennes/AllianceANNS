"""Validate metadata, phase semantics and planning without touching ANN inputs."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from benchlib.protocol import load_plans, merge, read_json, validate_profile, verify_sources, WORKLOADS

CONFIG = SCRIPTS / "configs/caselaw/profiles.json"


class CaseLawProfileTests(unittest.TestCase):
    def fixture(self, workload="origin"):
        catalog = read_json(CONFIG)
        profile = merge(catalog["common"], catalog["profiles"][workload])
        _, documents = verify_sources(CONFIG.parent, profile["sources"])
        return profile, documents

    def test_complete_frozen_grids_and_plan_counts(self):
        plans = load_plans(CONFIG)
        self.assertEqual([plan["workload"] for plan in plans], list(WORKLOADS))
        self.assertEqual([plan["formal_batches"] for plan in plans], [200, 300, 300, 300, 200])
        for plan in plans:
            self.assertFalse(plan["execution_supported"])
            self.assertFalse(plan["production_inputs_checked"])
            for case in plan["cases"]:
                self.assertEqual(len(case["efs"]), 20)
                self.assertEqual(case["repeats_per_ef"], 5)
                self.assertEqual(case["efs"], sorted(set(case["efs"])))
                if case["cache_policy"] == "equal":
                    self.assertEqual((plan["workload"], case["algorithm"]), ("p10", "allianceann"))

    def test_loading_cgroup_and_search_rss_are_distinct(self):
        profile, documents = self.fixture()
        self.assertEqual(profile["memory"]["loading"]["limit_bytes"], 9 << 30)
        self.assertEqual(profile["memory"]["search"]["limit_bytes"], 3 << 30)
        for field, value in (("enforcement", "cgroup_memory_max"), ("cgroup_memory_max", 3 << 30),
                             ("reset_peak_per_batch", False)):
            changed = copy.deepcopy(profile)
            changed["memory"]["search"][field] = value
            with self.assertRaises(ValueError):
                validate_profile(changed, documents)

    def test_cache_lifecycle_and_all_five_validation_are_mandatory(self):
        profile, documents = self.fixture()
        for key in ("dynamic_clear_per_query", "dynamic_clear_per_ef", "dynamic_clear_per_repeat", "empty_dynamic_before_timing"):
            changed = copy.deepcopy(profile)
            changed["cache_lifecycle"][key] = not changed["cache_lifecycle"][key]
            with self.assertRaises(ValueError):
                validate_profile(changed, documents)
        profile["aggregation"]["all_repeats_must_validate"] = False
        with self.assertRaises(ValueError):
            validate_profile(profile, documents)

    def test_frozen_budget_ef_and_route_mismatches_fail(self):
        for workload in WORKLOADS:
            profile, documents = self.fixture(workload)
            for change in (lambda p: p["cache_budgets"]["allianceann"].update(static_bytes=1),
                           lambda p: p["routing"].update(top_layer_postfilter=1-p["routing"]["top_layer_postfilter"]),
                           lambda p: p["efs"]["allianceann"].__setitem__(-1, p["efs"]["allianceann"][-1]+1)):
                changed = copy.deepcopy(profile)
                change(changed)
                with self.assertRaises(ValueError):
                    validate_profile(changed, documents)
        profile, documents = self.fixture("p10")
        profile["routing"]["prefilter_threshold"] = 8192
        with self.assertRaises(ValueError):
            validate_profile(profile, documents)
        profile, documents = self.fixture()
        del profile["request_bindings"]["efs/allianceann"]
        with self.assertRaises(ValueError):
            validate_profile(profile, documents)

    def test_paths_are_overridable_without_reading_production_files(self):
        plans = load_plans(CONFIG, ["p1"], {"data": "/nonexistent/caselaw", "experiments": "/nonexistent/drivers"})
        self.assertTrue(plans[0]["input_paths"]["query"].startswith("/nonexistent/caselaw/"))
        self.assertTrue(plans[0]["historical_driver_dependencies"][0]["path"].startswith("/nonexistent/drivers/"))
        with self.assertRaises(ValueError):
            load_plans(CONFIG, path_overrides={"unknown": "/tmp"})
        with self.assertRaises(ValueError):
            load_plans(CONFIG, path_overrides={"data": "relative/path"})

    def test_changed_metadata_hash_and_reference_escape_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content = b'{}'
            (root / "request.json").write_bytes(content)
            record = {"file": "request.json", "sha256": hashlib.sha256(content).hexdigest()}
            verify_sources(root, {"request": record})
            (root / "request.json").write_bytes(b'{"changed": true}')
            with self.assertRaises(ValueError):
                verify_sources(root, {"request": record})
            record["file"] = "../outside.json"
            with self.assertRaises(ValueError):
                verify_sources(root, {"request": record})

    def test_cli_is_read_only_and_has_no_run_mode(self):
        watched = (CONFIG.parent, SCRIPTS / "benchlib")
        before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for directory in watched for p in directory.rglob("*") if p.is_file()}
        result = subprocess.run([sys.executable, str(SCRIPTS / "plan_caselaw.py"), "--validate", "--workload", "p10"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["plans"][0]["formal_batches"], 200)
        rejected = subprocess.run([sys.executable, "-B", str(SCRIPTS / "plan_caselaw.py"), "--run"],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertNotEqual(rejected.returncode, 0)
        after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for directory in watched for p in directory.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
