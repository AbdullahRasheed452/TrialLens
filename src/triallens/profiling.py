# Peak memory measurement that works the same way on GPU and CPU.
#
# GPU: torch's allocator statistics, reset before the measured region, so the number is
# the extra memory the region needed on top of what was already allocated (weights).
# CPU: torch has no allocator statistics, so a background thread samples the process RSS
# every few milliseconds and reports the peak above the starting RSS. This is coarser
# (it includes allocator caching and misses spikes shorter than the sampling interval)
# and is reported as such.

from __future__ import annotations

import threading
import time

import psutil
import torch


class PeakMemory:
    def __init__(self, device, interval_s: float = 0.002):
        self.cuda = torch.device(device).type == "cuda"
        self.interval_s = interval_s
        self.peak_bytes = 0

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.base = torch.cuda.memory_allocated()
        else:
            self.proc = psutil.Process()
            self.base = self.proc.memory_info().rss
            self._max = self.base
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def _sample(self):
        while not self._stop.is_set():
            self._max = max(self._max, self.proc.memory_info().rss)
            time.sleep(self.interval_s)

    def __exit__(self, *exc):
        if self.cuda:
            torch.cuda.synchronize()
            self.peak_bytes = torch.cuda.max_memory_allocated() - self.base
        else:
            self._stop.set()
            self._thread.join()
            self._max = max(self._max, self.proc.memory_info().rss)
            self.peak_bytes = self._max - self.base
        return False

    @property
    def method(self) -> str:
        return "torch.cuda.max_memory_allocated" if self.cuda else "process RSS sampling"
