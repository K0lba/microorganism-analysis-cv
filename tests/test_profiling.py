import time

import numpy as np
import pytest

from microtrack.profiling import ResourceMonitor, rss_mb


def test_monitor_measures_time_and_memory_growth():
    with ResourceMonitor(interval=0.002) as m:
        big = np.ones(12_000_000, np.float64)          # ~96 MB, страницы реально выделены
        time.sleep(0.08)
        assert big.sum() > 0
        del big
    assert m.wall_s >= 0.08
    assert m.rss_delta_mb > 60
    assert m.rss_peak_mb >= m.rss_start_mb
    assert m.gpu_peak_mb is None or m.gpu_peak_mb >= 0


def test_monitor_small_block_has_small_delta():
    with ResourceMonitor() as m:
        x = sum(range(1000))
    assert x == 499500 and m.rss_delta_mb < 30
    assert rss_mb() > 0


def test_monitor_is_reusable_and_propagates_exceptions():
    m = ResourceMonitor()
    with pytest.raises(ValueError):
        with m:
            raise ValueError("boom")
    assert m.wall_s >= 0
    with m:
        pass
    assert m.wall_s >= 0
