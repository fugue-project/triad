from triad.utils.threading import RunOnce
import concurrent
from triad.utils.assertion import assert_or_throw
import cloudpickle


def test_runonce():
    def con(df, arr, n=1):
        arr[0] += n
        return df

    arr = [0]
    sc = RunOnce(con, lambda *args, **kwargs: id(args[0]))
    sc = cloudpickle.loads(cloudpickle.dumps(sc))
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
    sc = RunOnce(con, lambda *args, **kwargs: id(args[0]))
    test = [df1, df2] * 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(lambda df: assert_or_throw(sc(df, arr) == df, "failed"), test)
    assert 2 == arr[0]
