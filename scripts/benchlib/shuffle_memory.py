"""Shuffle-only RSS observation; no limits or process-state changes."""
from pathlib import Path
import re


def scope_memory_properties(memory):
    if memory is None:
        return ['-p', 'MemoryMax=infinity']
    if not isinstance(memory, int) or isinstance(memory, bool) or memory <= 0:
        raise ValueError('memory_max_bytes must be positive bytes or null')
    return ['-p', 'MemoryMax=' + str(memory), '-p', 'MemorySwapMax=0']


class ShuffleRssObservation:
    """Find the native child of this launch, then retain RSS/HWM across resets."""
    def __init__(self, parent, proc_root='/proc'):
        self.parent = parent
        self.root = Path(proc_root)
        self.tracked = None
        self.peak = 0
        self.samples = 0

    def sample(self):
        if self.tracked is None:
            pending = [self.parent]
            while pending:
                pid = pending.pop()
                try:
                    task = self.root / str(pid)
                    name = (task / 'comm').read_text().strip()
                    if name.startswith('alliance_shuffle') or name == 'alliance_shuffl':
                        self.tracked = pid
                        break
                    pending.extend(map(int, (task / 'task' / str(pid) / 'children').read_text().split()))
                except (OSError, ValueError):
                    continue
        if self.tracked is not None:
            try:
                status = (self.root / str(self.tracked) / 'status').read_text()
                values = re.findall(r'^Vm(?:RSS|HWM):\s+(\d+)\s+kB', status, re.M)
                if values:
                    self.peak = max(self.peak, max(map(int, values)) * 1024)
                    self.samples += 1
            except OSError:
                pass


def rss_measurements(text, time_rss_mib, observer):
    phases = [float(x) for x in re.findall(r'\[RSS\]\s+\S+\s+peak\s+([\d.]+) MB', text)]
    return dict(peak_rss_mib=max([time_rss_mib, observer.peak / (1 << 20)] + phases),
                time_peak_rss_mib=time_rss_mib, phase_peak_rss_mib=phases,
                sampled_rss_hwm_peak_bytes=observer.peak, rss_samples=observer.samples,
                rss_note='Maximum of phase VmHWM, 0.1s samples and GNU time; phase log rounds to 0.1 MiB.')
