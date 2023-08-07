import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa

from triad.utils.convert import to_size

from .iter import slice_iterable

T = TypeVar("T")


class BatchReslicer(Generic[T]):
    """Reslice batch streams with row or/and size limit

    :param row_limit: max row for each slice, defaults to None
    :param size_limit: max byte size for each slice, defaults to None

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
        """Get the number of rows and byte size of a batch

        :param batch: the batch object
        :return: the number of rows and byte size of the batch
        """
        raise NotImplementedError  # pragma: no cover

    def take(self, batch: T, start: int, length: int) -> T:
        """Take a slice of the batch

        :param batch: the batch object
        :param start: the start row index
        :param length: the number of rows to take

        :return: a slice of the batch
        """
        raise NotImplementedError  # pragma: no cover

    def concat(self, batches: List[T]) -> T:
        """Concatenate a list of batches into one batch

        :param batches: the list of batches
        :return: the concatenated batch
        """
        raise NotImplementedError  # pragma: no cover

    def reslice(self, batches: Iterable[T]) -> Iterable[T]:  # noqa: C901, A003
        """Reslice the batch stream into new batches constrained by the row or size limit

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

            if cache_rows >= row_limit:  # clean up edge cases
                yield self.concat(cache)
                cache = []
                cache_rows = 0

            if cache_rows + batch_rows < row_limit:
                cache.append(batch)
                cache_rows += batch_rows
            else:
                # here we guarantee initial_rows > 0
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


class PandasBatchReslicer(BatchReslicer[pd.DataFrame]):
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


class ArrowTableBatchReslicer(BatchReslicer[pa.Table]):
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


class NumpyArrayBatchReslicer(BatchReslicer[np.ndarray]):
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


class SortedBatchReslicer(Generic[T]):
    """Reslice batch streams (that are alredy sorted by keys) by keys.

    :param keys: group keys to reslice by
    """

    def __init__(
        self,
        keys: List[str],
    ) -> None:
        self._keys = keys
        self._last_row: Optional[np.ndarray] = None

    def take(self, batch: T, start: int, length: int) -> T:
        """Take a slice of the batch

        :param batch: the batch object
        :param start: the start row index
        :param length: the number of rows to take

        :return: a slice of the batch
        """
        raise NotImplementedError  # pragma: no cover

    def concat(self, batches: List[T]) -> T:
        """Concatenate a list of batches into one batch

        :param batches: the list of batches
        :return: the concatenated batch
        """
        raise NotImplementedError  # pragma: no cover

    def get_keys_ndarray(self, batch: T, keys: List[str]) -> np.ndarray:
        """Get the keys as a numpy array

        :param batch: the batch object
        :param keys: the keys to get

        :return: the keys as a numpy array
        """
        raise NotImplementedError  # pragma: no cover

    def get_batch_length(self, batch: T) -> int:
        """Get the number of rows in the batch

        :param batch: the batch object

        :return: the number of rows in the batch
        """
        raise NotImplementedError  # pragma: no cover

    def reslice(
        self, batches: Iterable[T]
    ) -> Iterable[Iterable[T]]:  # noqa: C901, A003
        """Reslice the batch stream into a stream of iterable of batches of the
        same keys

        :param batches: the batch stream

        :yield: an iterable of iterable of batches containing same keys
        """

        def slicer(
            n: int, current: Tuple[bool, T], last: Optional[Tuple[bool, T]]
        ) -> bool:
            return current[0]

        def get_slices() -> Iterable[Tuple[bool, T]]:
            for batch in batches:
                if self.get_batch_length(batch) > 0:
                    yield from self._reslice_single(batch)

        def transform(data: Iterable[Tuple[bool, T]]) -> Iterable[T]:
            for _, batch in data:
                yield batch

        for res in slice_iterable(get_slices(), slicer):
            yield transform(res)

    def reslice_and_merge(
        self, batches: Iterable[T]
    ) -> Iterable[T]:  # noqa: C901, A003
        """Reslice the batch stream into new batches, each containing the same keys

        :param batches: the batch stream

        :yield: an iterable of batches, each containing the same keys
        """

        cache: Optional[T] = None

        for batch in batches:
            if self.get_batch_length(batch) > 0:
                for diff, sub in self._reslice_single(batch):
                    if not diff:
                        cache = self.concat([cache, sub])  # type: ignore
                    else:
                        if cache is not None:
                            yield cache
                        cache = sub

        if cache is not None:
            yield cache

    def _reslice_single(self, batch: T) -> Iterable[Tuple[bool, T]]:
        a = self.get_keys_ndarray(batch, self._keys)
        b = np.roll(a, 1, axis=0)
        diff = self._diff(a, b)
        if self._last_row is not None:
            diff_from_last: bool = self._diff(a[0:1], self._last_row)[0]  # type: ignore
        else:
            diff_from_last = True
        self._last_row = a[-1:]
        points = np.where(diff)[0].tolist() + [a.shape[0]]
        if len(points) == 1:
            yield diff_from_last, batch
        else:
            for i in range(len(points) - 1):
                new_start = diff_from_last if i == 0 else True
                yield new_start, self.take(batch, points[i], points[i + 1] - points[i])

    def _diff(self, a: np.ndarray, b: np.ndarray) -> bool:
        return ((a == b) | ((a != a) & (b != b))).sum(axis=1) < len(self._keys)


class PandasSortedBatchReslicer(SortedBatchReslicer[pd.DataFrame]):
    def get_keys_ndarray(self, batch: pd.DataFrame, keys: List[str]) -> np.ndarray:
        return batch[keys].to_numpy()

    def get_batch_length(self, batch: pd.DataFrame) -> int:
        return batch.shape[0]

    def take(self, batch: pd.DataFrame, start: int, length: int) -> pd.DataFrame:
        return batch.iloc[start : start + length]

    def concat(self, batches: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(batches)


class ArrowTableSortedBatchReslicer(SortedBatchReslicer[pa.Table]):
    def get_keys_ndarray(self, batch: pa.Table, keys: List[str]) -> np.ndarray:
        return batch.select(keys).to_pandas().to_numpy()

    def get_batch_length(self, batch: pa.Table) -> int:
        return batch.num_rows

    def take(self, batch: pa.Table, start: int, length: int) -> pa.Table:
        return batch.slice(start, length)

    def concat(self, batches: List[pa.Table]) -> pa.Table:
        return pa.concat_tables(batches)
