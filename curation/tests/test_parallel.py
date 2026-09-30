"""test_parallel.py — unit tests for lib.parallel.parallel_map."""

import pytest

from lib.parallel import parallel_map

# fork() under pytest triggers a DeprecationWarning because pytest itself is
# multi-threaded. The real pipeline is single-threaded at fork time (verified:
# `python 03_filter.py --workers 4` with -W error::DeprecationWarning → clean).
pytestmark = pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded.*:DeprecationWarning"
)


def _square(x):
    return x * x


def _boom(x):
    if x == 2:
        raise ValueError("boom at 2")
    return x


def _echo(x):
    return x


class TestParallelMap:
    def test_serial_input_order(self):
        assert parallel_map(_square, [1, 2, 3], n_workers=1) == [1, 4, 9]

    def test_workers_match_serial(self):
        items = list(range(100))
        serial = parallel_map(_square, items, n_workers=1)
        assert parallel_map(_square, items, n_workers=4) == serial

    def test_identity_order_preserved(self):
        items = ["a", "b", "c", "d", "e"]
        assert parallel_map(_echo, items, n_workers=4) == items

    def test_empty_input(self):
        assert parallel_map(_square, [], n_workers=1) == []
        assert parallel_map(_square, [], n_workers=4) == []

    def test_more_workers_than_items(self):
        assert parallel_map(_square, [1, 2], n_workers=8) == [1, 4]

    def test_exception_propagates(self):
        with pytest.raises(ValueError, match="boom at 2"):
            parallel_map(_boom, [1, 2, 3], n_workers=4)
