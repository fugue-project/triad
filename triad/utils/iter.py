from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size

T = TypeVar("T")


def make_empty_aware(it: Union[Iterable[T], Iterator[T]]) -> "EmptyAwareIterable[T]":
    """Make an iterable empty aware, or return itself if already empty aware

    :param it: underlying iterable

    :return: EmptyAwareIterable[T]
    """
    return it if isinstance(it, EmptyAwareIterable) else EmptyAwareIterable(it)


def slice_iterable(
    it: Union[Iterable[T], Iterator[T]], slicer: Callable[[int, T, Optional[T]], bool]
) -> "Iterable[_SliceIterable[T]]":
    """Slice the original iterable into slices by slicer

    :param it: underlying iterable
    :param slicer: taking in current number, current value, last value,
        it decides if it's a new slice

    :yield: an iterable of iterables (_SliceIterable[T])
    """
    si = _SliceIterable(it, slicer)
    while si._state < 3:
        yield si
        si.recycle()


def to_kv_iterable(  # noqa: C901
    data: Any, none_as_empty: bool = True
) -> Iterable[Tuple[Any, Any]]:
    """Convert data to iterable of key value pairs

    :param data: input object, it can be a dict or Iterable[Tuple[Any, Any]]
        or Iterable[List[Any]]
    :param none_as_empty: if to treat None as empty iterable

    :raises ValueError: if input is None and `none_as_empty==False`
    :raises TypeError or ValueError: if input data type is not acceptable

    :yield: iterable of key value pair as tuples
    """
    if data is None:
        assert_or_throw(none_as_empty, ValueError("data can't be None"))
    elif isinstance(data, Dict):
        for k, v in data.items():
            yield k, v
    elif isinstance(data, Iterable):
        ei = make_empty_aware(data)
        if not ei.empty:
            first = ei.peek()
            if isinstance(first, tuple):
                for k, v in ei:
                    yield k, v
            elif isinstance(first, List):
                for arr in ei:
                    if len(arr) == 2:
                        yield arr[0], arr[1]
                    else:
                        raise TypeError(f"{arr} is not an acceptable item")
            else:
                raise TypeError(f"{first} is not an acceptable item")
    else:
        raise TypeError(f"{type(data)} is not supported")


class EmptyAwareIterable(Iterable[T]):
    """A wrapper of iterable that can tell if the underlying
    iterable is empty, it can also peek a non-empty iterable.

    :param it: the underlying iterable

    :raises StopIteration: raised by the underlying iterable
    """

    def __init__(self, it: Union[Iterable[T], Iterator[T]]):
        self._last: Optional[T] = None
        if not isinstance(it, Iterator):
            self._iter = iter(it)
        else:
            self._iter = it
        self._state = 0
        self._fill_last()

    @property
    def empty(self) -> bool:
        """Check if the underlying iterable has more items

        :return: whether it is empty
        """
        return self._fill_last() >= 2

    def peek(self) -> T:
        """Return the next of the iterable without moving

        :raises StopIteration: if it's empty
        :return: the `next` item
        """
        if not self.empty:
            return self._last  # type: ignore
        raise StopIteration("Can't peek empty iterable")

    def __iter__(self) -> Any:
        """Wrapper of the underlying __iter__

        :yield: next object
        """
        while self._fill_last() < 2:
            self._state = 0
            yield self._last

    def _fill_last(self) -> int:
        try:
            if self._state == 0:  # last is not filled
                self._last = next(self._iter)
                self._state = 1  # last filed not used
        except StopIteration:
            self._state = 3  # end
        return self._state


class Slicer(object):
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
        sizer: Optional[Callable[[Any], int]] = None,  # func for getsizeof(item)
        row_limit: Optional[int] = None,
        size_limit: Any = None,
        slicer: Optional[Callable[[int, T, Optional[T]], bool]] = None,
    ) -> None:
        self._sizer = sizer
        self._slicer = slicer
        if row_limit is None:
            self._row_limit = 0
        else:
            self._row_limit = row_limit
        if size_limit is None:
            self._size_limit = 0
        else:
            self._size_limit = to_size(str(size_limit))
        assert (
            self._size_limit == 0 or self._sizer is not None
        ), "sizer must be set when size_limit>0"
        self._current_row = 1
        self._current_size = 0

    def slice(  # noqa C901
        self, orig_it: Iterable[T]
    ) -> Iterable[EmptyAwareIterable[T]]:
        """Slice the original iterable into slices by the combined slicing logic

        :param orig_it: ther original iterable

        :yield: an iterable of EmptyAwareIterable
        """
        it = make_empty_aware(orig_it)
        if it.empty:
            pass
        elif self._row_limit <= 0 and self._size_limit <= 0:
            if self._slicer is None:
                yield it
            else:
                for _slice in slice_iterable(it, self._slicer):
                    yield _slice
        elif self._row_limit > 0 and self._size_limit <= 0:
            if self._slicer is None:
                for _slice in slice_iterable(it, self._is_boundary_row_only):
                    yield _slice
            else:
                for _slice in slice_iterable(it, self._is_boundary_row_only_w_slicer):
                    yield _slice
        else:
            self._current_size = self._sizer(it.peek())  # type: ignore
            self._current_row = 1
            if self._row_limit <= 0 and self._size_limit > 0:
                for _slice in slice_iterable(it, self._is_boundary_size_only):
                    yield _slice
            else:
                for _slice in slice_iterable(it, self._is_boundary):
                    yield _slice

    def _is_boundary_row_only(self, no: int, current: Any, last: Any) -> bool:
        return no % self._row_limit == 0

    def _is_boundary_row_only_w_slicer(self, no: int, current: Any, last: Any) -> bool:
        # self._slicer must be invoked even hitting row limit
        is_boundary = self._slicer is not None and self._slicer(no, current, last)
        if self._current_row >= self._row_limit or is_boundary:
            self._current_row = 1
            return True
        self._current_row += 1
        return False

    def _is_boundary_size_only(self, no: int, current: Any, last: Any) -> bool:
        obj_size = self._sizer(current)  # type: ignore
        next_size = self._current_size + obj_size
        # self._slicer must be invoked even hitting size limit
        is_boundary = self._slicer is not None and self._slicer(no, current, last)
        if next_size > self._size_limit or is_boundary:
            self._current_size = obj_size
            return True
        self._current_size = next_size
        return False

    def _is_boundary(self, no: int, current: Any, last: Any) -> bool:
        obj_size = self._sizer(current)  # type: ignore
        next_size = self._current_size + obj_size
        # self._slicer must be invoked even hitting row limit and size limit
        is_boundary = self._slicer is not None and self._slicer(no, current, last)
        if (
            next_size > self._size_limit
            or self._current_row >= self._row_limit  # noqa: W503
            or is_boundary  # noqa: W503
        ):
            self._current_size = obj_size
            self._current_row = 1
            return True
        self._current_size = next_size
        self._current_row += 1
        return False


class _SliceIterable(EmptyAwareIterable[T]):
    def __init__(self, it: Union[Iterable[T], Iterator[T]], slicer: Any):
        self._n = 0
        self._slicer = slicer
        super().__init__(it)

    def recycle(self) -> None:
        if self._state < 2:
            for _ in self:
                pass
        if self._state == 2:
            self._state = 1

    def _fill_last(self) -> int:
        try:
            if self._state == 0:  # last is not filled
                last = self._last
                self._last = next(self._iter)
                is_boundary = self._n > 0 and self._slicer(self._n, self._last, last)
                self._n += 1
                self._state = 2 if is_boundary else 1  # last filed not used
        except StopIteration:
            self._state = 3  # end
        return self._state
