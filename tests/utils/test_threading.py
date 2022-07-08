import concurrent

import cloudpickle
from triad.utils.assertion import assert_or_throw
from triad.utils.threading import SerializableRLock, run_once


def test_serializable_rlock():
    lock = cloudpickle.loads(cloudpickle.dumps(SerializableRLock()))

    arr = [0]

    def add(n):
        with lock:
            arr[0] += n

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(add, [1] * 100)

    assert 100 == arr[0]


def test_runonce():
    @run_once
    def f1(a, b):
        """doc"""
        arr[0] += 1
        return a + b

    arr = [0]
    sc = cloudpickle.loads(cloudpickle.dumps(f1))
    assert 3 == f1(1, 2)
    assert 3 == f1(1, 2)
    assert 4 == f1(2, 2)
    assert 2 == arr[0]
    assert "doc" == f1.__doc__

    @run_once(key_func=lambda *args, **kwargs: True)
    def f2(a, b):
        """doc"""
        arr[0] += 1
        return a + b

    arr = [0]
    sc = cloudpickle.loads(cloudpickle.dumps(f2))
    assert 3 == f2(1, 2)
    assert 3 == f2(1, 2)
    assert 3 == f2(2, 2)
    assert 1 == arr[0]
    assert "doc" == f1.__doc__

    @run_once(key_func=lambda *args, **kwargs: id(args[0]))
    def con(df, arr, n=1):
        arr[0] += n
        return df

    arr = [0]
    sc = cloudpickle.loads(cloudpickle.dumps(con))
    df1 = [[0]]
    df2 = [[0]]
    assert sc(df1, arr) is df1
    assert 1 == arr[0]
    assert sc(df1, arr) is df1
    assert 1 == arr[0]
    assert sc(df2, arr, n=2) is df2
    assert 3 == arr[0]
    assert sc(df2, arr, n=3) is df2
    assert 3 == arr[0]
    arr[0] = 0
    # multi thread test

    test = [df1, df2] * 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(lambda df: assert_or_throw(con(df, arr) == df, "failed"), test)
    assert 2 == arr[0]
