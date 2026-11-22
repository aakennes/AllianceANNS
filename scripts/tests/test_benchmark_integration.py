"""End-to-end archive, independent replay, paired export, and resume contracts."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchlib import runner
from benchlib.spec import digest_file
from benchlib.validation import preflight_curve
from test_benchmark_validation import Fixture


class IntegrationTests(unittest.TestCase):
    def test_all_native_archive_adapters_through_real_verifier_and_export(self):
        for algorithm in ("allianceann", "pipewow", "pipeannfilter"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as temporary:
                root = Path(temporary).resolve()
                fixture = Fixture(root / "fixture", algorithm)
                curve = fixture.curve
                curve.update(cache_policy="landing" if algorithm == "allianceann" else "native", repeats=5,
                             scope_prefix={"allianceann": "caselaw-rss4-r12-", "pipewow": "caselaw-pw-rss4-",
                                           "pipeannfilter": "caselaw-paf-origin-rss3-"}[algorithm])
                fixture.measured.update(repeat=1, requested_repeats=5)
                fixture.save()
                for repeat in range(2, 6):
                    target = fixture.root / "raw" / ("r" + str(repeat)) / "ef16"
                    shutil.copytree(str(fixture.directory), str(target))
                    measured = dict(fixture.measured, repeat=repeat)
                    (target / "measured.json").write_text(json.dumps(measured), encoding="utf-8")
                child = root / "copy_archive.py"
                child.write_text("from pathlib import Path\nimport shutil,sys\n"
                                 "source,output,counter=map(Path,sys.argv[1:])\n"
                                 "with counter.open('a') as stream: stream.write('one native invocation\\n')\n"
                                 "for item in source.iterdir(): shutil.copytree(str(item),str(output/item.name))\n"
                                 "print('SEARCH_VERIFIED fingerprint=12345')\n", encoding="utf-8")
                python = str(Path(sys.executable).resolve())
                counter = root / "calls.txt"
                curve.update(argv=[python, str(child), str(fixture.root / "raw"), "{output}", str(counter)],
                             env={"OMP_NUM_THREADS": "1", "CASELAW_OUTPUT": "{output}"})
                paths = [python, str(child)] + list(curve["inputs"].values())
                paths += [str(p) for p in (fixture.root / "raw").rglob("*") if p.is_file()]
                spec = {"schema_version": 2, "id": "synthetic-integration-" + algorithm,
                        "memory": fixture.spec["memory"], "aggregation": {"policy": "qps-paired-trim", "repeats": 5},
                        "inputs": [{"path": p, "sha256": digest_file(p)} for p in paths],
                        "curves": [curve], "lock_paths": [str(root / "shared.lock")]}
                self.assertEqual(preflight_curve(spec, curve)["status"], "ready")

                def execute(plan, case, directory):
                    argv = [v.replace("{output}", str(directory / "raw")) for v in case["argv"]]
                    with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
                        result = subprocess.run(argv, stdout=out, stderr=err)
                    sent = runner._read(directory / "sent.json")
                    unit = case["scope_prefix"] + sent["run_token"][:20] + ".scope"
                    # A fixture executor supplies synthetic scope evidence here;
                    # the separate opt-in systemd smoke reads real kernel files.
                    runner._write(directory / "memory.json", {
                        "schema_version": 1, "status": "finished", "exit_code": result.returncode,
                        "request_sha256": sent["request_sha256"], "curve_id": case["id"], "run_token": sent["run_token"],
                        "scope_unit": unit, "cgroup_path": "/sys/fs/cgroup/fixture/" + unit,
                        "initial_memory_max": str(plan["memory"]["loading_limit_bytes"]), "final_memory_max": "max",
                        "initial_memory_swap_max": "0", "final_memory_swap_max": "0",
                        "affinity_before": [1], "affinity_after": [1],
                        "memory_events_before": {"oom": 0, "oom_kill": 0}, "memory_events_after": {"oom": 0, "oom_kill": 0}})
                    return {"exit_code": result.returncode}

                job = root / "job"
                completed = runner.run_experiment(spec, job, executor=execute)
                self.assertEqual((completed["status"], completed["row_count"]), ("verified", 5))
                exports = json.loads((job / "aggregate_export/export_manifest.json").read_text())
                self.assertTrue(all(digest_file(job / "aggregate_export" / name) == digest for name, digest in exports.items()))
                decision = json.loads((job / "aggregate_export/trim_decisions.json").read_text())[0]
                self.assertEqual(decision["kept_repeat_ids"], [2, 3, 4])
                runner.run_experiment(spec, job, resume=True, executor=lambda *args: self.fail("resume relaunched native measurement"))
                self.assertEqual(len(counter.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
