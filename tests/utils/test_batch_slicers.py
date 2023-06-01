from typing import Tuple

import pandas as pd

from triad.utils.batch_slicers import PandasBatchSlicer


class _MockSlicer(PandasBatchSlicer):
    def get_rows_and_size(self, batch: pd.DataFrame) -> Tuple[int, int]:
        return len(batch), len(batch) * 1024  # assume each row is 1KB


def test_base_batch_slicer():
    def _test(chunks, row_limit, size_limit, expected):
        slicer = _MockSlicer(row_limit=row_limit, size_limit=size_limit)
        orig = []
        start = 0
        for x in chunks:
            orig.append(pd.DataFrame({"a": range(start, start + x)}))
            start += x
        dfs = list(slicer.slice(orig))
        assert sum(chunks) == sum(len(t) for t in dfs)
        if len(dfs) > 0:
            assert pd.concat(dfs).values.tolist() == pd.concat(orig).values.tolist()
            assert list(len(t) for t in slicer.slice(orig)) == expected

    _test([], None, None, [])
    _test([0], 0, 0, [])
    _test([0, 1, 0, 2], -1, 0, [1, 2])

    _test([], 2, 0, [])
    _test([0], 2, 0, [])
    _test([1], 1, 0, [1])
    _test([1], 2, 0, [1])
    _test([2], 2, 0, [2])
    _test([1, 1], 2, 0, [2])
    _test([2, 2], 1, 0, [1, 1, 1, 1])
    _test([2, 1], 2, 0, [2, 1])
    _test([1, 0, 2], 2, 0, [2, 1])
    _test([1, 2, 3], 2, 0, [2, 2, 2])
    _test([1, 2, 3], 10, 0, [6])
    _test([1, 2, 3], 1, 0, [1, 1, 1, 1, 1, 1])

    _test([], 0, "2k", [])
    _test([0], 0, "2k", [])
    _test([1], 0, 1024, [1])
    _test([1], 0, 1000, [1])
    _test([1], 0, 2048, [1])
    _test([2], 0, "2k", [2])
    _test([1, 1], 0, 1030, [1, 1])
    _test([2], 0, 1030, [1, 1])
    _test([1, 1], 0, "2k", [2])
    _test([2, 2], 0, "1k", [1, 1, 1, 1])
    _test([2, 1], 0, "2k", [2, 1])
    _test([1, 0, 2], 0, "2k", [2, 1])
    _test([1, 2, 3], 0, "2k", [2, 2, 2])
    _test([1, 2, 3], 0, "10m", [6])
    _test([1, 2, 3], 0, "1k", [1, 1, 1, 1, 1, 1])
    _test([1, 2, 3], 0, 1000, [1, 1, 1, 1, 1, 1])

    _test([1, 1], 3, 1030, [1, 1])
    _test([1, 2, 3], 1, "5m", [1, 1, 1, 1, 1, 1])
