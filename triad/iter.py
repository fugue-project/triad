from typing import Iterator, Iterable, Any, Optional, Union, TypeVar, Callable
from triad.convert import to_size

T = TypeVar("T")


class EmptyAwareIterable(Iterable[T]):
    """A wrapper of iterable that can tell if the underlying
    iterable is empty, it can also peek a non-empty iterable.

    Args:
        it (Union[Iterable[T], Iterator[T]]: the underlying iterable

    Raises:
        StopIteration: raised by the underlying iterable
    """

    def __init__(self, it: Union[Iterable[T], Iterator[T]]):
        self._last: Optional[T] = None
        if not isinstance(it, Iterator):
            self._iter = iter(it)
        else:
            self._iter = it
        self._state = 0
        self._fill_last()

    def empty(self) -> bool:
        """Check if the underlying iterable has more items

        Returns:
            bool: whether it is empty
        """
        return self._fill_last() >= 2

    def peek(self) -> T:
        """Return the next of the iterable without moving

        Raises:
            StopIteration: if it's empty

        Returns:
            Optional[T]: [description]
        """
        if not self.empty():
            return self._last  # type: ignore
        raise StopIteration("Can't peek empty iterable")

    def __iter__(self) -> Any:
        """Wrapper of the underlying __iter__

        Yields:
            Any: next object
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


def make_empty_aware(it: Union[Iterable[T], Iterator[T]]) -> EmptyAwareIterable[T]:
    """Make an iterable empty aware, or return itself if already empty aware

    Args:
        it (Union[Iterable[T], Iterator[T]]): underlying iterable

    Returns:
        EmptyAwareIterable[T]: wrapper iterable
    """
    return it if isinstance(it, EmptyAwareIterable) else EmptyAwareIterable(it)


def slice_iterable(
    it: Union[Iterable[T], Iterator[T]], slicer: Callable[[int, T, Optional[T]], bool]
) -> "Iterable[_SliceIterable[T]]":
    """Slice the original iterable into slices by slicer

    Args:
        it (Union[Iterable[T], Iterator[T]]): underlying iterable
        slicer (Callable[[int, T, Optional[T]], bool]): taking in current number,
        current value, last value, it decides if it's a new slice

    Yields:
        Iterable[_SliceIterable[T]]: an iterable of iterables
    """
    si = _SliceIterable(it, slicer)
    while si._state < 3:
        yield si
        si.recycle()

        """
        :param sizer: func for getsizeof(each_item)
        :param row_limit: max rows for each slice, None or <=0 for no limit
        :param size_limit: max size for each slice, None or <=0 for no limit
        :param slicer: custom is_boundary function, will only be called
            if not exceeding row_limit and size_limit
        """


class Slicer(object):
    """A better version of :func:`~triad.iter.slice_iterable`

    Args:
        sizer (Any): [description]
        row_limit (int, optional): Max row for each slice. Defaults to None,
        size_limit (Any, optional): Max byte size for each slice (can
            be a size expression, see :func:`~triad.convert.to_size`). Defaults to None.
        slicer (Callable[[int, T, Optional[T]], bool], optional): [description].
            Defaults to None.
    """

    def __init__(
        self,
        sizer: Any,  # func for getsizeof(item)
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
        self._current_row = 1
        self._current_size = 0

    def split(self, orig_it: Iterable[T]) -> Iterable[Iterable[T]]:  # noqa C901
        it = make_empty_aware(orig_it)
        if it.empty():
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
            self._current_size = self._sizer(it.peek())
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
        if self._current_row >= self._row_limit or (
            self._slicer is not None and self._slicer(no, current, last)
        ):
            self._current_row = 1
            return True
        self._current_row += 1
        return False

    def _is_boundary_size_only(self, no: int, current: Any, last: Any) -> bool:
        obj_size = self._sizer(current)
        next_size = self._current_size + obj_size
        if next_size > self._size_limit or (
            self._slicer is not None and self._slicer(no, current, last)
        ):
            self._current_size = obj_size
            return True
        self._current_size = next_size
        return False

    def _is_boundary(self, no: int, current: Any, last: Any) -> bool:
        obj_size = self._sizer(current)
        next_size = self._current_size + obj_size
        if (
            next_size > self._size_limit
            or self._current_row >= self._row_limit
            or (self._slicer is not None and self._slicer(no, current, last))
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
