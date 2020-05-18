import itertools
from collections import OrderedDict

from pytest import raises
from triad.utils.iter import (
    EmptyAwareIterable,
    Slicer,
    make_empty_aware,
    slice_iterable,
    to_kv_iterable,
)


def test_empty_aware_iterable():
    i = _get_iterable("1,2,3")
    e = make_empty_aware(i)
    assert not e.empty
    assert "1,2,3" == ",".join(e)
    assert e.empty

    i = _get_iterable("1")
    e = EmptyAwareIterable(i)
    assert not e.empty
    assert not e.empty
    assert "1" == ",".join(e)
    assert e.empty

    e = EmptyAwareIterable([])
    assert e.empty
    assert "" == ",".join(e)
    assert e.empty
    raises(StopIteration, lambda: e.peek())

    i = _get_iterable("1,2,3")
    e = EmptyAwareIterable(i)
    assert not e.empty
    assert "1,2" == ",".join(itertools.islice(e, 2))
    assert not e.empty
    assert "3" == ",".join(itertools.islice(e, 2))
    assert e.empty

    i = _get_iterable("1,2,3")
    e = EmptyAwareIterable(iter(i))
    assert not e.empty
    assert "1" == e.peek()
    assert "1,2" == ",".join(itertools.islice(e, 2))
    assert not e.empty
    assert "3" == e.peek()
    assert "3" == ",".join(itertools.islice(e, 2))
    assert e.empty


def test_empty_aware_iterable_recursive():
    i = _get_iterable("1,2,3")
    e = make_empty_aware(i)
    ee = make_empty_aware(_wrap_iterable(e, True))
    assert "1,2,3" == ",".join(ee)
    i = _get_iterable("1,2,3")
    e = make_empty_aware(i)
    ee = make_empty_aware(_wrap_iterable(e, False))
    assert "1t,2t,3t" == ",".join(ee)


def test_to_kv_iterable():
    data1 = [(1, 1), (2, 2)]
    data2 = OrderedDict(data1)
    data3 = [[1, 1], (2, 2)]
    data4 = [[1, 1, 3], (2, 2)]
    data5 = [1, (2, 2)]
    data6 = [(1, 1), (2, 2, 3)]

    assert [] == list(to_kv_iterable(None, none_as_empty=True))
    assert [] == list(to_kv_iterable(None))
    assert [] == list(to_kv_iterable([]))
    raises(ValueError, lambda: list(to_kv_iterable(None, none_as_empty=False)))
    assert data1 == list(to_kv_iterable(data1))
    assert data1 == list(to_kv_iterable(data2))
    assert data1 == list(to_kv_iterable(data3))
    raises(TypeError, lambda: list(to_kv_iterable(data4)))
    raises(TypeError, lambda: list(to_kv_iterable(data5)))
    raises(TypeError, lambda: list(to_kv_iterable(123)))
    raises(ValueError, lambda: list(to_kv_iterable(data6)))


def test_slice_iterable():
    # make sure empty iterable will yield no slice
    ll = list(slice_iterable([], lambda n, c, l: n % 2 == 0))
    assert 0 == len(ll)
    assert_slice("", [], lambda n, c, l: n % 2 == 0, lambda x: x)
    assert_slice("1,2-3,4-5", range(1, 6), lambda n, c, l: n % 2 == 0, lambda x: x)
    assert_slice("1,2,3,4,5", range(1, 6), lambda n, c, l: c < l, lambda x: x)
    assert_slice("1-2-3-4-5", range(1, 6), lambda n, c, l: c > l, lambda x: x)
    # for each slice, only iterate some of them
    assert_slice(
        "1-3-5",
        range(1, 6),
        lambda n, c, l: n % 2 == 0,
        lambda x: itertools.islice(x, 1),
    )
    assert_slice(
        "1,2-3,4-5",
        range(1, 6),
        lambda n, c, l: n % 2 == 0,
        lambda x: itertools.islice(x, 100),
    )
    assert_slice(
        "--", range(1, 6), lambda n, c, l: n % 2 == 0, lambda x: itertools.islice(x, 0)
    )
    n = -1

    def sl(it):
        nonlocal n
        n += 1
        return itertools.islice(it, n)

    assert_slice("-3-5", range(1, 6), lambda n, c, l: n % 2 == 0, sl)


def test_slicer():
    assert_slicer("", [], 1, 0, lambda x: 1)
    assert_slicer("", [], 0, 0, lambda x: 1)
    assert_slicer("", [], 0, 1, lambda x: 1)
    assert_slicer("", [], 1, 1, lambda x: 1)
    assert_slicer(".000", [1, 1, 1], 0, 0, None)
    assert_slicer(".000", [1, 1, 1], None, None, None)
    assert_slicer(".0.1.2", [1, 1, 1], 1, 0, None)
    assert_slicer(".00.1", [1, 1, 1], 2, 0, None)
    assert_slicer(".0.1.2", [1, 1, 1], 0, 1, lambda x: 1)
    assert_slicer(".0.1.2", [1, 1, 1], 0, 1, lambda x: 10)
    assert_slicer(".00.1", [1, 1, 1], 0, 2, lambda x: 1)
    assert_slicer(".0.1.2", [1, 1, 1], 1, 2, lambda x: 1)
    assert_slicer(".00.1", [1, 1, 1], 10, 2, lambda x: 1)
    assert_slicer(".000", [1, 1, 1], 10, 20, lambda x: 1)
    assert_slicer(".0.1.2", [1, 1, 1], 1, "2k", lambda x: 1)
    assert_slicer(".00.1", [1, 1, 1], None, "2k", lambda x: 1024)

    class C(object):
        def __init__(self):
            self.arr = []

        def c(self, no, current, last):
            self.arr.append([current, last])
            return current > last

    c = C()
    assert_slicer("", [], 1, 0, lambda x: 1, c.c)
    assert_slicer("", [], 0, 0, lambda x: 1, c.c)
    assert_slicer("", [], 0, 1, lambda x: 1, c.c)
    assert_slicer("", [], 1, 1, lambda x: 1, c.c)
    assert 0 == len(c.arr)
    assert_slicer(".000", [1, 1, 1], 0, 0, None, c.c)
    c = C()
    assert_slicer(".0.1.2", [1, 2, 3], 0, 0, None, c.c)
    assert [[2, 1], [3, 2]] == c.arr
    c = C()
    assert_slicer(".0.1.2", [1, 0, -1], 1, 0, None, c.c)
    assert [[0, 1], [-1, 0]] == c.arr  # is_boundary must be called anyway
    c = C()
    assert_slicer(".00.1", [1, 0, -1], 2, 0, None, c.c)
    assert [[0, 1], [-1, 0]] == c.arr  # is_boundary must be called anyway
    c = C()
    # size and row counters should reset after slicer taking effect
    assert_slicer(".0.11", [1, 2, 1], 2, 0, None, c.c)
    assert [[2, 1], [1, 2]] == c.arr
    c = C()
    assert_slicer(".00.1", [1, 0, -1], 0, 2, lambda x: 1, c.c)
    assert [[0, 1], [-1, 0]] == c.arr
    c = C()
    assert_slicer(".0.1.2", [1, 1, 1], 1, 2, lambda x: 1, c.c)
    assert [[1, 1], [1, 1]] == c.arr
    c = C()
    # size and row counters should reset after slicer taking effect
    assert_slicer(".0.11", [1, 2, 1], 10, 2, lambda x: 1, c.c)
    assert [[2, 1], [1, 2]] == c.arr
    assert_slicer(".000", [1, 1, 1], 10, 20, lambda x: 1)
    assert_slicer(".0.1.2", [1, 1, 1], 1, "2k", lambda x: 1)
    assert_slicer(".00.1", [1, 1, 1], None, "2k", lambda x: 1024)


def assert_slice(expected, iterable, slicer, slice_proc):
    ll = []
    for x in slice_iterable(iterable, slicer):
        assert not x.empty
        assert isinstance(x, EmptyAwareIterable)
        ll.append(",".join(map(str, slice_proc(x))))
    s = "-".join(ll)
    assert expected == s


def assert_slicer(expected, arr, max_row, max_size, sizer, slicer=None):
    r = []
    n = 0
    c = Slicer(sizer, max_row, max_size, slicer=slicer)
    for chunk in c.slice(arr):
        assert isinstance(chunk, EmptyAwareIterable)
        r.append(".")
        for x in chunk:
            r.append(str(n))
        n += 1
    assert expected == "".join(r)


def _get_iterable(s):
    for ss in s.split(","):
        yield ss


def _make_iterable(n):
    while n > 0:
        yield n
        n -= 1


def _wrap_iterable(it, p):
    if p:
        for x in it:
            yield x
    else:
        for x in it:
            yield str(x) + "t"
