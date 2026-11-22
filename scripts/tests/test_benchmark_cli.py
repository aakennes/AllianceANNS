"""The public CLI defaults to a read-only plan and rejects incomplete bindings."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CLI = Path(__file__).resolve().parents[1] / "benchmark.py"


class CLITests(unittest.TestCase):
    def test_default_plan_does_not_need_production_files_or_write_a_job(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as temporary:
            result = subprocess.run([sys.executable, "-B", str(CLI), "--workload", "p10", "--algorithm", "allianceann"],
                                    cwd=temporary, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout)
            self.assertEqual(output["status"], "planned")
            self.assertFalse(output["production_inputs_checked"])
            self.assertEqual(output["formal_queries_run"], 0)
            self.assertEqual(len(output["spec"]["curves"]), 2)
            self.assertTrue(all(len(c["efs"]) == 20 and c["repeats"] == 5 for c in output["spec"]["curves"]))
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_incomplete_identity_and_ambiguous_execution_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as temporary:
            result = subprocess.run([sys.executable, "-B", str(CLI), "--workload", "p1", "--algorithm", "pipeannfilter", "--validate"],
                                    cwd=temporary, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("incomplete runtime bindings", result.stderr)
            for args in (["--run"], ["--resume"], ["--job", str(Path(temporary) / "unrequested")]):
                result = subprocess.run([sys.executable, "-B", str(CLI)] + args, cwd=temporary,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                self.assertEqual(result.returncode, 2)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
