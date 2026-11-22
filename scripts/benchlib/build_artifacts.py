"""Build/shuffle lifecycle helpers. Never delete artifacts or run ANN queries."""
from pathlib import Path
import json
import os
import stat


def disk_snapshot(root):
    """Count allocated bytes as well as logical lengths; ignore links and races."""
    root = Path(root)
    files, identities = [], set()
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(parent) / name).is_symlink()]
        for name in names:
            path = Path(parent) / name
            try:
                entry = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(entry.st_mode):
                continue
            identity = (entry.st_dev, entry.st_ino)
            if identity in identities:
                continue
            identities.add(identity)
            allocated = entry.st_blocks * 512 if hasattr(entry, 'st_blocks') else None
            files.append(dict(path=str(path.relative_to(root)), logical_bytes=entry.st_size,
                              allocated_bytes=allocated))
    known = all(row['allocated_bytes'] is not None for row in files)
    disk = os.statvfs(root) if hasattr(os, 'statvfs') else None
    return dict(logical_bytes=sum(row['logical_bytes'] for row in files),
                allocated_bytes=sum(row['allocated_bytes'] for row in files) if known else None,
                available_bytes=disk.f_bavail * disk.f_frsize if disk else None, files=files)


class DiskObservation:
    """Stream compact samples; retain file details only for observed maxima."""
    def __init__(self, root, name):
        self.root = Path(root)
        self.path = self.root / (name + '.disk.jsonl')
        self.stream = self.path.open('x', encoding='utf-8')
        self.count = 0
        self.peak_logical = None
        self.peak_allocated = None

    def sample(self, elapsed):
        value = dict(elapsed_seconds=elapsed, **disk_snapshot(self.root))
        self.count += 1
        if self.peak_logical is None or value['logical_bytes'] > self.peak_logical['logical_bytes']:
            self.peak_logical = value
        if value['allocated_bytes'] is not None and (self.peak_allocated is None or
                value['allocated_bytes'] > self.peak_allocated['allocated_bytes']):
            self.peak_allocated = value
        compact = {k: v for k, v in value.items() if k != 'files'}
        self.stream.write(json.dumps(compact) + '\n')
        self.stream.flush()

    def close(self):
        self.stream.close()
        return dict(samples_file=self.path.name, sample_count=self.count,
                    observed_peak_logical=self.peak_logical, observed_peak_allocated=self.peak_allocated,
                    note='One-second samples of this output directory including logs; not an exact continuous peak or other filesystems.')


def finalization_argv(python, finalizer, verifier, root, parameters):
    """Reuse the existing verified close-resume operation without wildcard cleanup."""
    if parameters.get('baseatt') in (None, '', 'serial'):
        raise ValueError('automatic finalization requires an explicit attributes file')
    return [str(python), str(finalizer), '--artifact-root', str(root),
            '--graph', parameters['graph_file'], '--meta', parameters['meta_file'],
            '--base', parameters['basevec'], '--format', parameters.get('basevec_format', 'fvecs'),
            '--attrs', parameters['baseatt'], '--space', parameters.get('space', 'l2'),
            '--verifier', str(verifier), '--apply', '--close-resume-window']
