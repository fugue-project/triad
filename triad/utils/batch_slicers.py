import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa

from triad.utils.convert import to_size

T = TypeVar("T")


class BatchSlicer(Generic[T]):
    """A better version of :func:`~triad.iter.slice_iterable`

    :param sizer: the function to get size of an item
    :param row_limit: max row for each slice, defaults to None
    :param size_limit: max byte size for each slice, defaults to None
    :param slicer: taking in current number, current value, last value,
        it decides if it's a new slice

    :raises AssertionError: if `size_limit` is not None but `sizer` is None
    """

    def __init__(
        self,
        row_limit: Optional[int] = None,
        size_limit: Any = None,
    ) -> None:
        if row_limit is None:
            self._row_limit = 0
        else:
            self._row_limit = row_limit
        if size_limit is None:
            self._size_limit = 0
        else:
            self._size_limit = to_size(str(size_limit))

    def get_rows_and_size(self, batch: T) -> Tuple[int, int]:
        raise NotImplementedError  # pragma: no cover

    def take(self, batch: T, start: int, length: int) -> T:
        raise NotImplementedError  # pragma: no cover

    def concat(self, batches: List[T]) -> T:
        raise NotImplementedError  # pragma: no cover

    def slice(self, batches: Iterable[T]) -> Iterable[T]:  # noqa: C901, A003
        """Slice the batch stream into new batches constrained by the row or size limit

        :param batches: the batch stream

        :yield: an iterable of batches of the same type with the constraints
        """
        if self._row_limit <= 0 and self._size_limit <= 0:
            for batch in batches:
                batch_rows, _ = self.get_rows_and_size(batch)
                if batch_rows > 0:
                    yield batch
            return
        cache: List[T] = []
        total_rows, total_size = 0, 0
        cache_rows = 0
        for batch in batches:
            batch_rows, batch_size = self.get_rows_and_size(batch)
            if batch_rows == 0:
                continue
            total_rows += batch_rows
            total_size += batch_size
            row_size = total_size / total_rows

            if self._row_limit > 0 and self._size_limit <= 0:
                row_limit = self._row_limit
            elif self._row_limit <= 0 and self._size_limit > 0:
                row_limit = max(math.floor(self._size_limit / row_size), 1)
            else:
                row_limit = min(
                    self._row_limit, max(math.floor(self._size_limit / row_size), 1)
                )

            if cache_rows + batch_rows < row_limit:
                cache.append(batch)
                cache_rows += batch_rows
            else:
                slices, remain = self._slice_rows(
                    batch_rows, row_limit - cache_rows, slice_rows=row_limit
                )
                for i, rg in enumerate(slices):
                    chunk = self.take(batch, rg[0], rg[1])
                    if i == 0:
                        yield self.concat(cache + [chunk])
                        cache = []
                        cache_rows = 0
                    else:
                        yield chunk
                if remain[1] > 0:
                    cache.append(self.take(batch, remain[0], remain[1]))
                    cache_rows += remain[1]
        if len(cache) > 0:
            yield self.concat(cache)

    def _slice_rows(
        self, batch_rows: int, initial_rows: int, slice_rows: int
    ) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
        start = 0
        if initial_rows >= batch_rows:
            return [(0, batch_rows)], (0, 0)
        slices = [(0, initial_rows)]
        start = initial_rows
        while True:
            if batch_rows - start < slice_rows:
                return slices, (start, batch_rows - start)
            slices.append((start, slice_rows))
            start += slice_rows


class PandasBatchSlicer(BatchSlicer[pd.DataFrame]):
    def get_rows_and_size(self, batch: pd.DataFrame) -> Tuple[int, int]:
        return batch.shape[0], batch.memory_usage(deep=True).sum()

    def take(self, batch: pd.DataFrame, start: int, length: int) -> pd.DataFrame:
        if start == 0 and length == batch.shape[0]:
            return batch
        return batch.iloc[start : start + length]

    def concat(self, batches: List[pd.DataFrame]) -> pd.DataFrame:
        if len(batches) == 1:
            return batches[0]
        return pd.concat(batches)


class ArrowTableBatchSlicer(BatchSlicer[pa.Table]):
    def get_rows_and_size(self, batch: pa.Table) -> Tuple[int, int]:
        return batch.num_rows, batch.nbytes

    def take(self, batch: pa.Table, start: int, length: int) -> pa.Table:
        if start == 0 and length == batch.num_rows:
            return batch
        return batch.slice(start, length)

    def concat(self, batches: List[pa.Table]) -> pa.Table:
        if len(batches) == 1:
            return batches[0]
        return pa.concat_tables(batches)


class NumpyArrayBatchSlicer(BatchSlicer[np.ndarray]):
    def get_rows_and_size(self, batch: np.ndarray) -> Tuple[int, int]:
        return batch.shape[0], batch.nbytes

    def take(self, batch: np.ndarray, start: int, length: int) -> np.ndarray:
        if start == 0 and length == batch.shape[0]:
            return batch
        return batch[start : start + length]

    def concat(self, batches: List[np.ndarray]) -> np.ndarray:
        if len(batches) == 1:
            return batches[0]
        return np.concatenate(batches, axis=0)
