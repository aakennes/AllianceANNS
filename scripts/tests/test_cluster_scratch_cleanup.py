import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'cleanup_cluster_scratch.py'
spec = importlib.util.spec_from_file_location('cleanup_cluster_scratch', MODULE)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


@unittest.skipUnless(sys.platform.startswith('linux'), 'requires Linux proc/inode ownership')
class ScratchCleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.graph = self.root / 'graph.bin'
        self.graph.write_bytes(b'graph')
        self.generation = self.root / 'graph.bin.cluster-scratch.ABC123'
        self.generation.mkdir(mode=0o700)
        child = subprocess.Popen([sys.executable, '-c', 'pass'])
        self.dead_pid = child.pid
        child.wait()
        source = self.graph.stat()
        files = []
        for name in ('input.records', 'cluster.0', 'cluster.1'):
            path = self.generation / name
            path.write_bytes(b'owned')
            entry = path.stat()
            files.append(dict(name=name, dev=entry.st_dev, ino=entry.st_ino, allow_absent=True))
        self.manifest = dict(version=1, kind='alliance-cluster-spill', pid=self.dead_pid,
                             boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                             start_ticks='0', graph=str(self.graph), graph_dev=source.st_dev,
                             graph_ino=source.st_ino, files=files)
        self.save()

    def save(self):
        (self.generation / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_default_read_only_and_apply_exact_files(self):
        report = cleanup.inspect_or_remove(self.graph, self.generation)
        self.assertEqual(report['action'], 'would_remove')
        self.assertEqual(report['files'], 3)
        self.assertTrue(self.generation.exists())
        report = cleanup.inspect_or_remove(self.graph, self.generation, True)
        self.assertEqual(report['action'], 'remove')
        self.assertFalse(self.generation.exists())
        self.assertEqual(self.graph.read_bytes(), b'graph')

    def test_already_consumed_files_are_explicitly_allowed(self):
        (self.generation / 'input.records').unlink()
        report = cleanup.inspect_or_remove(self.graph, self.generation, True)
        self.assertEqual(report['already_consumed'], 1)
        self.assertFalse(self.generation.exists())

    def test_active_owner_refused(self):
        process = Path('/proc/self/stat').read_text()
        self.manifest['pid'] = os.getpid()
        self.manifest['start_ticks'] = process[process.rindex(')') + 2:].split()[19]
        self.save()
        with self.assertRaisesRegex(ValueError, 'still active'):
            cleanup.inspect_or_remove(self.graph, self.generation, True)
        self.assertTrue((self.generation / 'cluster.0').exists())

    def test_replaced_inode_refused_before_any_deletion(self):
        replacement = self.generation / 'replacement'
        replacement.write_bytes(b'other')
        replacement.replace(self.generation / 'cluster.1')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            cleanup.inspect_or_remove(self.graph, self.generation, True)
        self.assertTrue((self.generation / 'cluster.0').exists())
        self.assertEqual((self.generation / 'cluster.1').read_bytes(), b'other')

    def test_unexpected_file_refused(self):
        (self.generation / 'unrelated').write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError, 'unexpected or missing'):
            cleanup.inspect_or_remove(self.graph, self.generation, True)
        self.assertTrue((self.generation / 'cluster.0').exists())

    def test_missing_manifest_refused(self):
        (self.generation / 'manifest.json').unlink()
        with self.assertRaises(FileNotFoundError):
            cleanup.inspect_or_remove(self.graph, self.generation, True)
        self.assertTrue((self.generation / 'cluster.0').exists())

    def test_symlink_owned_file_refused(self):
        (self.generation / 'cluster.0').unlink()
        (self.generation / 'cluster.0').symlink_to(self.graph)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            cleanup.inspect_or_remove(self.graph, self.generation, True)
        self.assertEqual(self.graph.read_bytes(), b'graph')


if __name__ == '__main__':
    unittest.main()
