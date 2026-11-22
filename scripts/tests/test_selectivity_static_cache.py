import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "prepare_selectivity_static_cache.py"
spec = importlib.util.spec_from_file_location("prepare_cache", SCRIPT)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("graph", "meta", "sidecar"):
            (self.root / name).write_bytes(name.encode())
        header = [0x314D4C4C, 2, 10, 128, 5, 32, 10, 5, 4096, 3, 1, 1]
        (self.root / "ml").write_bytes(struct.pack("<12Q", *header))
        (self.root / "ranges").write_bytes(struct.pack("<4i", 0, 9, 1, 8))
        self.calls = self.root / "calls"
        fake = '''#!/usr/bin/env python3
import json,sys,struct
from pathlib import Path
a=dict(zip(sys.argv[1::2],sys.argv[2::2]))
calls=Path(__file__).resolve().parents[1]/'calls'
with calls.open('a') as f:f.write(Path(__file__).name+'\\n')
if 'collect' in Path(__file__).name:
 counts=[0,0,2] if a['--filter_mode']!='prefilter' else [0,0,0]
 Path(a['--out_file']).write_bytes(struct.pack('<4Q',3,*counts))
 Path(a['--report_file']).write_text(json.dumps(dict(histogram=counts,graph_queries=sum(counts),prefilter_queries=2-sum(counts))))
else:
 for b in map(int,a['--budgets'].split(',')):
  p=Path(a['--out_prefix']+'_'+str(b)+'.layout');p.write_bytes(b'cache'+str(b).encode()+str('--static_landing_file' in a).encode())
  print(f'# static_prepared: path={p} budget_bytes={b} resident_bytes={b-1} payload_bytes={b-2} cached_nodes=1')
'''
        for name in ("alliance_collect_landing", "alliance_prepare_static_cache"):
            p = self.bin / name
            p.write_text(fake)
            p.chmod(0o755)
        self.base = []
        for option, value in dict(graph_file=self.root / "graph", meta_file=self.root / "meta",
                                  ml_graph_file=self.root / "ml", ml_meta_file=self.root / "sidecar",
                                  query_rng=self.root / "ranges", cache_dir=self.root / "cache",
                                  bin_dir=self.bin, budgets="1000").items():
            self.base += ["--" + option, str(value)]

    def run_cache(self, *extra):
        return cache.prepare(cache.parser().parse_args(self.base + list(extra)))

    def call_count(self, kind):
        return self.calls.read_text().count(kind) if self.calls.exists() else 0

    def test_default_equal_and_explicit_false_ignore_landing(self):
        first = self.run_cache("--static_landing_file", "/nonexistent")
        second = self.run_cache("--static_cache_use_landing", "false", "--static_landing_query_rng", "/bad")
        self.assertEqual(first, second)
        self.assertEqual(first["mode"], "equal")
        self.assertFalse((self.root / "cache/landing").exists())
        self.assertEqual(self.call_count("collect"), 0)

    def test_reuse_and_content_invalidation_and_budget_reuse(self):
        first = self.run_cache("--static_cache_use_landing", "true")
        self.assertEqual(first, self.run_cache("--static_cache_use_landing", "true"))
        larger = self.run_cache("--static_cache_use_landing", "true", "--budgets", "2000")
        self.assertEqual(first["histogram"], larger["histogram"])
        self.assertEqual(self.call_count("collect"), 1)
        self.assertEqual(self.call_count("prepare_static"), 2)
        (self.root / "ranges").write_bytes(struct.pack("<4i", 0, 8, 1, 7))
        changed = self.run_cache("--static_cache_use_landing", "true")
        self.assertNotEqual(first["histogram"], changed["histogram"])
        self.assertNotEqual(first["sidecars"], changed["sidecars"])
        self.assertEqual(self.call_count("collect"), 2)

    def test_independent_training_range_and_external_combined_histogram(self):
        train = self.root / "training"
        train.write_bytes(struct.pack("<2i", 0, 9))
        first = self.run_cache("--static_cache_use_landing", "true", "--static_landing_query_rng", str(train))
        (self.root / "ranges").write_bytes(struct.pack("<2i", 4, 5))
        second = self.run_cache("--static_cache_use_landing", "true", "--static_landing_query_rng", str(train))
        self.assertEqual(first, second)
        combined = self.root / "combined.landing"
        combined.write_bytes(struct.pack("<4Q", 3, 5, 4, 3))
        result = self.run_cache("--static_cache_use_landing", "true", "--static_landing_file", str(combined))
        self.assertEqual(result["mode"], "landing")
        self.assertEqual(self.call_count("collect"), 1)
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertEqual(manifest["histogram"], [5, 4, 3])

    def test_modes_separate_and_corruption_is_preserved_and_repaired(self):
        equal = self.run_cache()
        weighted = self.run_cache("--static_cache_use_landing", "true")
        self.assertNotEqual(equal["sidecars"], weighted["sidecars"])
        path = Path(weighted["sidecars"]["1000"])
        path.write_bytes(b"corrupt")
        fixed = self.run_cache("--static_cache_use_landing", "true")
        self.assertEqual(fixed, weighted)
        self.assertNotEqual(path.read_bytes(), b"corrupt")
        self.assertTrue(list(path.parent.parent.glob("*.invalid-*")))

    def test_all_prefilter_explicit_equal_fallback(self):
        result = self.run_cache("--static_cache_use_landing", "true", "--filter_mode", "prefilter")
        self.assertEqual(result["mode"], "equal_no_graph_landings")
        self.assertEqual(result["routing"]["prefilter_queries"], 2)

    def test_dry_run_does_not_touch_inputs_or_cache(self):
        self.assertTrue(self.run_cache("--dry_run")["dry_run"])
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.root / "cache").exists())

    def test_external_histogram_rejects_truncated_or_bad_shape(self):
        external = self.root / "bad"
        for data in (b"", struct.pack("<Q", 100)):
            external.write_bytes(data)
            with self.assertRaises(ValueError):
                self.run_cache("--static_cache_use_landing", "true", "--static_landing_file", str(external))

    def test_concurrent_requests_prepare_once(self):
        command = [sys.executable, str(SCRIPT), *self.base, "--static_cache_use_landing", "true"]
        procs = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
        results = []
        for proc in procs:
            out, err = proc.communicate(timeout=30)
            self.assertEqual(proc.returncode, 0, err.decode())
            results.append(json.loads(out))
        self.assertEqual(results[0], results[1])
        self.assertEqual(self.call_count("collect"), 1)
        self.assertEqual(self.call_count("prepare_static"), 1)


class ShellTests(unittest.TestCase):
    def test_dry_run_selectivities_and_explicit_false(self):
        shell = SCRIPT.parent / "disk_search_common.sh"
        for dataset in ("arxiv", "sift1m"):
            command = ["bash", str(shell), dataset, "--dry_run", "--inputs_dir", "/inputs",
                       "--ml_graph_file", "/layout", "--static_cache_budget", "1000",
                       "--static_landing_file", "/bad", "--static_cache_use_landing", "false",
                       "--selectivities", "p1,p4", "--efs", "20,30", "--k", "10"]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("_p1.ranges.ibin", result.stdout)
            self.assertIn("_p4.k10.gt.ivecs", result.stdout)
            self.assertIn("--static_cache_use_landing false", result.stdout)
            self.assertNotIn("--static_landing_file /bad", result.stdout)
            self.assertEqual(result.stdout.count("--print_sidecar"), 2)


if __name__ == "__main__":
    unittest.main()
