from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

from triad.utils.batch_reslicers import (
    ArrowTableBatchReslicer,
    NumpyArrayBatchReslicer,
    PandasBatchReslicer,
    PandasSortedBatchReslicer,
    ArrowTableSortedBatchReslicer,
)


class _MockSlicer(PandasBatchReslicer):
    def get_rows_and_size(self, batch: pd.DataFrame) -> Tuple[int, int]:
        return len(batch), len(batch) * 1024  # assume each row is 1KB


def test_base_batch_reslicer():
    def _test(chunks, row_limit, size_limit, expected):
        slicer = _MockSlicer(row_limit=row_limit, size_limit=size_limit)
        orig = []
        start = 0
        for x in chunks:
            orig.append(pd.DataFrame({"a": range(start, start + x)}))
            start += x
        dfs = list(slicer.reslice(orig))
        assert sum(chunks) == sum(len(t) for t in dfs)
        if len(dfs) > 0:
            assert pd.concat(dfs).values.tolist() == pd.concat(orig).values.tolist()
            assert list(len(t) for t in dfs) == expected

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


def test_pandas_reslicer():
    s = PandasBatchReslicer()
    df = pd.DataFrame({"a": range(10)})
    assert s.get_rows_and_size(df) == (10, df.memory_usage(deep=True).sum())
    assert s.take(df, 0, 10) is df
    assert s.take(df, 1, 5).values.tolist() == df.iloc[1:6].values.tolist()
    assert s.concat([df]) is df
    assert s.concat([df, df]).values.tolist() == pd.concat([df, df]).values.tolist()

    # edge case: the adjusted row limit is smaller or equal to cache rows
    dfs1 = [pd.DataFrame({"a": ["abc"] * 100})] * 1
    dfs2 = [pd.DataFrame({"a": ["abc" * 100] * 10})] * 10
    dfs = dfs1 + dfs2 + dfs1
    s = PandasBatchReslicer(size_limit=dfs1[0].memory_usage(deep=True).sum() + 200)
    chunks = list(s.reslice(dfs))
    assert all(len(x) > 0 for x in chunks)
    assert pd.concat(chunks).values.tolist() == pd.concat(dfs).values.tolist()


def test_arrow_reslicer():
    s = ArrowTableBatchReslicer()
    pdf = pd.DataFrame({"a": range(10)})
    df = pa.Table.from_pandas(pdf)
    assert s.get_rows_and_size(df) == (10, df.nbytes)
    assert s.take(df, 0, 10) is df
    assert s.take(df, 1, 5).to_pandas().values.tolist() == pdf.iloc[1:6].values.tolist()
    assert s.concat([df]) is df
    assert (
        s.concat([df, df]).to_pandas().values.tolist()
        == pd.concat([pdf, pdf]).values.tolist()
    )


def test_numpy_reslicer():
    s = NumpyArrayBatchReslicer()
    pdf = pd.DataFrame({"a": range(10)})
    df = pdf.to_numpy()
    assert s.get_rows_and_size(df) == (10, df.nbytes)
    assert s.take(df, 0, 10) is df
    assert (s.take(df, 1, 5) == np.arange(1, 6)[:, None]).all()
    assert s.concat([df]) is df
    assert (s.concat([df, df]) == pd.concat([pdf, pdf]).to_numpy()).all()


def test_sorted_pandas_reslicer():
    def _make_df(data):
        for x in data:
            yield pd.DataFrame(
                {
                    "a": x,
                    "b": pd.Series([None] * len(x), dtype="object"),
                    "c": range(len(x)),
                }
            )

    def _test(data, expected, mode="first"):
        s = PandasSortedBatchReslicer(keys=["a", "b"])
        actual = []
        for dfs in s.reslice(_make_df(data)):
            if mode == "first":
                df = next(dfs)
                actual.append(str(df.iloc[0]["a"]) + "-" + str(df.iloc[0]["c"]))
            elif mode == "all":
                dfs = list(dfs)
                actual.append(str(dfs[0].iloc[0]["a"]) + "-" + str(len(pd.concat(dfs))))
        assert actual == expected
        s = PandasSortedBatchReslicer(keys=["a", "b"])
        actual = []
        for df in s.reslice_and_merge(_make_df(data)):
            if mode == "first":
                actual.append(str(df.iloc[0]["a"]) + "-" + str(df.iloc[0]["c"]))
            elif mode == "all":
                actual.append(str(df.iloc[0]["a"]) + "-" + str(len(df)))
        assert actual == expected

    _test([], [])
    _test([[]], [])
    _test([[1, 1]], ["1-0"])
    _test([[1, 1, 2]], ["1-0", "2-2"])

    _test([[1, 1], [], [1, 1], []], ["1-0"])
    _test([[1, 1], [], [1, 2], []], ["1-0", "2-1"])
    _test([[1, 1], [], [2, 2], []], ["1-0", "2-0"])
    _test([[], [1, 2], [], [2, 2], []], ["1-0", "2-1"])
    _test([[], [1], [1], [2], [2]], ["1-0", "2-0"])

    _test([[], [1], [1, 1, 2], [], [2], [], [2]], ["1-3", "2-3"], mode="all")


def test_sorted_arrow_reslicer():
    def _make_df(data):
        for x in data:
            yield pa.Table.from_pandas(
                pd.DataFrame(
                    {
                        "a": x,
                        "b": pd.Series([None] * len(x), dtype="object"),
                        "c": range(len(x)),
                    }
                )
            )

    def _test(data, expected, mode="first"):
        s = ArrowTableSortedBatchReslicer(keys=["a", "b"])
        actual = []
        for dfs in s.reslice(_make_df(data)):
            if mode == "first":
                df = next(dfs).to_pandas()
                actual.append(str(df.iloc[0]["a"]) + "-" + str(df.iloc[0]["c"]))
            elif mode == "all":
                dfs = [x.to_pandas() for x in dfs]
                actual.append(str(dfs[0].iloc[0]["a"]) + "-" + str(len(pd.concat(dfs))))
        assert actual == expected
        s = ArrowTableSortedBatchReslicer(keys=["a", "b"])
        actual = []
        for df in s.reslice_and_merge(_make_df(data)):
            df = df.to_pandas()
            if mode == "first":
                actual.append(str(df.iloc[0]["a"]) + "-" + str(df.iloc[0]["c"]))
            elif mode == "all":
                actual.append(str(df.iloc[0]["a"]) + "-" + str(len(df)))
        assert actual == expected

    _test([], [])
    _test([[]], [])
    _test([[1, 1]], ["1-0"])
    _test([[1, 1, 2]], ["1-0", "2-2"])

    _test([[1, 1], [], [1, 1], []], ["1-0"])
    _test([[1, 1], [], [1, 2], []], ["1-0", "2-1"])
    _test([[1, 1], [], [2, 2], []], ["1-0", "2-0"])
    _test([[], [1, 2], [], [2, 2], []], ["1-0", "2-1"])
    _test([[], [1], [1], [2], [2]], ["1-0", "2-0"])

    _test([[], [1], [1, 1, 2], [], [2], [], [2]], ["1-3", "2-3"], mode="all")
