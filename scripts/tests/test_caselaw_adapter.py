"""Native argv/identity adapter tests; only tiny synthetic assets are accessed."""

import hashlib
import json
import os
from pathlib import Path
import posixpath
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from benchlib.caselaw import build_experiment
from benchlib.protocol import read_json
from benchlib.spec import validate_spec, fingerprint_inputs

CONFIG = SCRIPTS / "configs/caselaw/profiles.json"
BINDINGS = SCRIPTS / "configs/caselaw/runtime.json"


def synthetic_bindings(directory):
    """Relocate explicitly fake files; do not read or validate production assets."""
    runtime = read_json(BINDINGS)
    runtime["evidence_files"] = []
    root = Path(directory)

    def relocate(path):
        parent = hashlib.sha256(posixpath.dirname(path).encode()).hexdigest()[:12]
        return (root / "assets" / parent / posixpath.basename(path)).as_posix()

    for asset in runtime["assets"].values():
        original = asset["path"]
        path = Path(relocate(original))
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = ("synthetic test asset: " + original).encode()
        path.write_bytes(contents)
        asset.update(path=path.as_posix(), sha256=hashlib.sha256(contents).hexdigest(), identity_mode="sha256")
        asset.pop("expected_stat", None)
        asset.pop("availability", None)
    for binding in runtime["curves"].values():
        binding["binding_errors"] = []
        if "index_prefix" in binding:
            binding["index_prefix"] = relocate(binding["index_prefix"])
    runtime["lock_paths"] = [(root / "locks" / (str(i) + ".lock")).as_posix() for i in range(4)]
    target = root / "runtime.json"
    target.write_text(json.dumps(runtime), encoding="utf-8")
    return target, runtime


class CaseLawAdapterTests(unittest.TestCase):
    def test_native_commands_match_formal_command_metadata(self):
        spec = build_experiment(CONFIG, BINDINGS)
        evidence = read_json(BINDINGS.parent / "runtime_evidence.json")["documents"]
        self.assertEqual(len(spec["curves"]), 13)
        for curve in spec["curves"]:
            original = evidence[curve["source_command"]["path"]]["data"]
            previous = original["native_argv"]
            if curve["algorithm"] == "allianceann":
                self.assertEqual(curve["argv"][0], previous[0])
                actual = dict(zip(curve["argv"][1::2], curve["argv"][2::2]))
                expected = dict(zip(previous[1::2], previous[2::2]))
                expected.setdefault("--top_layer_postfilter", "1")
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual) * 2 + 1, len(curve["argv"]))
            else:
                normalized = list(previous)
                if curve["algorithm"] == "pipeannfilter":
                    normalized[7] = "{output}"
                self.assertEqual(curve["argv"], normalized)
            for key in ("CASELAW_REPEATS", "CASELAW_EXPECTED_FP", "CASELAW_EXPECTED_BITS", "CASELAW_EXPECTED_NQ",
                        "CASELAW_SEARCH_RSS_BYTES", "CASELAW_LOADING_RSS_BYTES", "CASELAW_STATIC_BUDGET"):
                if curve["algorithm"] == "pipeannfilter" and key == "CASELAW_EXPECTED_FP":
                    self.assertNotIn(key, curve["env"])
                    continue
                self.assertEqual(curve["env"][key], str(original["native_environment"][key]))

    def test_alliance_only_is_complete_and_does_not_inherit_baseline_errors(self):
        spec = build_experiment(CONFIG, BINDINGS, algorithms=["allianceann"])
        validate_spec(spec)
        self.assertEqual(len(spec["curves"]), 6)
        self.assertEqual(spec["binding_errors"], [])
        self.assertTrue(all(asset.get("identity_mode") != "unbound" for asset in spec["inputs"]))
        self.assertTrue(all(asset.get("availability") != "known_missing" for asset in spec["inputs"]))
        stat_assets = [asset for asset in spec["inputs"] if asset.get("sha256") is None]
        self.assertEqual(len(stat_assets), 3)
        for asset in stat_assets:
            self.assertEqual(asset["role"], "index")
            self.assertEqual(asset["identity_mode"], "stat_then_sha256")
            self.assertEqual(len(asset["expected_stat"]), 3)
            self.assertTrue(asset["provenance"]["sha256"])

    def test_missing_paf_identity_is_explicit_and_blocks_strict_validation(self):
        spec = build_experiment(CONFIG, BINDINGS, algorithms=["pipeannfilter"])
        self.assertTrue(spec["binding_errors"])
        self.assertTrue(any(asset.get("availability") == "known_missing" for asset in spec["inputs"]))
        self.assertTrue(any(asset.get("identity_mode") == "unbound" for asset in spec["inputs"]))
        self.assertTrue(all("CASELAW_EXPECTED_FP" not in curve["env"] for curve in spec["curves"]))
        with self.assertRaisesRegex(ValueError, "incomplete runtime bindings"):
            validate_spec(spec)

    def test_filtering_preserves_exact_frozen_p10_graph_protocol(self):
        spec = build_experiment(CONFIG, BINDINGS, workloads=["p10"], cache_policies=["equal"])
        self.assertEqual(len(spec["curves"]), 1)
        curve = spec["curves"][0]
        args = dict(zip(curve["argv"][1::2], curve["argv"][2::2]))
        self.assertEqual(args["--prefilter_threshold"], "1")
        self.assertEqual(args["--top_layer_postfilter"], "0")
        self.assertEqual(curve["algorithm"], "allianceann")
        self.assertEqual(len(curve["efs"]), 20)
        self.assertEqual(curve["repeats"], 5)
        self.assertTrue(curve["expected"]["require_zero_prefilter_queries"])
        self.assertEqual(spec["memory"], {"loading_limit_bytes": 9 << 30, "search_rss_limit_bytes": 3 << 30})

    def test_synthetic_bindings_relocate_inputs_without_reading_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, runtime = synthetic_bindings(temporary)
            asset_root = (Path(temporary) / "assets").resolve()
            native_open = Path.open

            def metadata_only_open(target, *args, **kwargs):
                self.assertNotIn(asset_root, target.resolve().parents, "adapter opened a native input")
                return native_open(target, *args, **kwargs)

            with patch.object(Path, "open", metadata_only_open):
                spec = build_experiment(CONFIG, path)
            validate_spec(spec)
            self.assertEqual(len(spec["curves"]), 13)
            self.assertFalse(spec["production_inputs_checked"])
            self.assertTrue(all(asset_root in Path(curve["binary_path"]).resolve().parents for curve in spec["curves"]))

    def test_deleted_synthetic_baseline_fails_before_any_hashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, runtime = synthetic_bindings(temporary)
            spec = build_experiment(CONFIG, path, workloads=["origin"], algorithms=["pipewow"])
            pages = next(asset["path"] for asset in spec["inputs"] if asset["path"].endswith("index.pages"))
            Path(pages).unlink()
            with patch("benchlib.spec.digest_file", side_effect=AssertionError("must inspect existence before hashing")):
                with self.assertRaisesRegex(ValueError, "required asset is missing"):
                    fingerprint_inputs(spec)

    def test_no_unrecognized_placeholders_or_extra_repeat_processes(self):
        spec = build_experiment(CONFIG, BINDINGS)
        for curve in spec["curves"]:
            self.assertEqual(curve["env"]["CASELAW_OUTPUT"], "{output}")
            self.assertEqual(curve["env"]["CASELAW_REPEATS"], "5")
            self.assertEqual(curve["argv"][0], curve["binary_path"])
            for text in curve["argv"] + list(curve["env"].values()):
                self.assertNotIn("{", text.replace("{output}", ""))
            self.assertNotIn("systemd-run", curve["argv"])
            self.assertNotIn("python", curve["argv"][0])

    def test_invalid_selections_and_changed_profile_hash_fail(self):
        for kwargs in ({"algorithms": ["unknown"]}, {"cache_policies": []},
                       {"workloads": ["p1"], "cache_policies": ["equal"]}):
            with self.assertRaises(ValueError):
                build_experiment(CONFIG, BINDINGS, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            path, runtime = synthetic_bindings(temporary)
            runtime["profiles_sha256"] = "0" * 64
            path.write_text(json.dumps(runtime), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different frozen profile"):
                build_experiment(CONFIG, path)

    def test_experiment_identity_is_stable_across_hash_seeds(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "from benchlib.caselaw import build_experiment; "
                "print(build_experiment(sys.argv[2], sys.argv[3])['id'])")
        identities = []
        for seed in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            result = subprocess.run([sys.executable, "-B", "-c", code, str(SCRIPTS), str(CONFIG), str(BINDINGS)],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            identities.append(result.stdout.strip())
        self.assertEqual(identities[0], identities[1])


if __name__ == "__main__":
    unittest.main()
