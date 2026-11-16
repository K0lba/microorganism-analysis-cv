"""Замер времени и памяти (RSS процесса и пик GPU) для участка кода."""
from __future__ import annotations

import threading
import time
from typing import Optional

try:                                    # psutil даёт пик внутри вызова; без него — грубая оценка
    import psutil
except ImportError:                     # pragma: no cover
    psutil = None

_MB = 2 ** 20


def rss_mb() -> float:
    """Текущий RSS процесса, МиБ (без psutil — пик за всё время жизни процесса)."""
    if psutil is not None:
        return psutil.Process().memory_info().rss / _MB
    import resource                     # pragma: no cover
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _torch_cuda():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


class ResourceMonitor:
    """Контекстный менеджер: время, пик RSS и пик памяти GPU внутри блока.

    >>> with ResourceMonitor() as m: work()
    >>> m.wall_s, m.rss_delta_mb, m.gpu_peak_mb
    """

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.wall_s = 0.0
        self.rss_start_mb = 0.0
        self.rss_peak_mb = 0.0
        self.rss_end_mb = 0.0
        self.gpu_peak_mb: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def rss_delta_mb(self) -> float:
        """Насколько пик RSS в блоке превысил RSS на входе (>= 0)."""
        return max(0.0, self.rss_peak_mb - self.rss_start_mb)

    def _sample(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            self.rss_peak_mb = max(self.rss_peak_mb, proc.memory_info().rss / _MB)
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        torch = _torch_cuda()
        if torch is not None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.rss_start_mb = self.rss_peak_mb = rss_mb()
        self._stop.clear()
        if psutil is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        torch = _torch_cuda()
        if torch is not None:
            torch.cuda.synchronize()
            self.gpu_peak_mb = torch.cuda.max_memory_allocated() / _MB
        self.wall_s = time.perf_counter() - self._t0
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.rss_end_mb = rss_mb()
        self.rss_peak_mb = max(self.rss_peak_mb, self.rss_end_mb)
