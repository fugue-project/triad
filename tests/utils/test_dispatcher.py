from typing import Any

from pytest import raises
from triad.utils.dispatcher import (
    conditional_broadcaster,
    conditional_dispatcher,
    run_at_def,
)
from pkg_resources import EntryPoint, get_distribution


def test_run_at_def():
    data = []

    @run_at_def
    def f1():
        """test1"""
        data.append(1)

    assert [1] == data
    assert "test1" == f1.__doc__

    @run_at_def()
    def f2():
        data.append(1)

    assert [1, 1] == data

    @run_at_def(x=2)
    def f3(x) -> int:
        """test3"""
        data.append(x)

    assert [1, 1, 2] == data
    assert "test3" == f3.__doc__

    @run_at_def()
    def f4(a=3) -> int:
        """test4"""
        data.append(a)

    assert [1, 1, 2, 3] == data
    assert "test4" == f4.__doc__


def test_conditional_dispatcher():
    @conditional_dispatcher
    def f1(a: Any, b: int) -> int:
        """test1"""
        raise NotImplementedError

    assert "test1" == f1.__doc__

    @conditional_dispatcher(entry_point="invalid")
    def f2(a: Any, b: int) -> int:
        """test2"""
        raise NotImplementedError

    assert "test2" == f2.__doc__

    def _bad_matcher(a, *args, **kwargs):
        raise Exception

    f1.register(
        matcher=lambda a, *args, **kwargs: isinstance(a, str),
        func=lambda a, b: len(a) + b,
    )
    f1.register(
        matcher=_bad_matcher,  # this should be ignored
        func=lambda a, b: len(a) + b,
    )
    f1.register(
        matcher=lambda a, *args, **kwargs: isinstance(a, str),
        func=lambda a, b: len(a) + b + 5,
    )

    @f1.candidate(lambda a, b: isinstance(a, int))
    def f3(a: int, b: int):
        return a + b

    f1.register(
        matcher=lambda a, *args, **kwargs: isinstance(a, str),
        func=lambda a, b: len(a) + b - 1,
        priority=2,
    )

    assert 5 == f1(2, 3)
    # higher priority
    assert 6 == f1("abcd", 3)

    f1.register(
        matcher=lambda a, *args, **kwargs: isinstance(a, str),
        func=lambda a, b: len(a) + b - 2,
        priority=2,
    )

    # newer registration
    assert 5 == f1("abcd", 3)

    with raises(NotImplementedError):
        f1([0], True)

    with raises(NotImplementedError):
        f2(True, True)


def test_conditional_broadcaster():
    @conditional_broadcaster
    def f1(a=1) -> int:
        """test1"""
        raise NotImplementedError

    assert "test1" == f1.__doc__

    @conditional_broadcaster(entry_point="invalid")
    def f2(a: Any) -> int:
        """test2"""
        raise NotImplementedError

    assert "test2" == f2.__doc__

    data = []

    @f1.candidate(lambda x: x < 10, priority=1)
    def f11(a):
        data.append(a + 1)

    @f1.candidate(lambda x: x < 10, priority=2)
    def f12(a):
        data.append(a + 2)

    f1(5)
    assert [7, 6] == data

    with raises(NotImplementedError):
        f1(11)  # no matching

    with raises(NotImplementedError):
        f2(10)  # no registration


def test_preload(mocker):
    mocker.patch(
        "triad.utils.dispatcher._entry_points",
        return_value={
            "tests.plugins": [
                EntryPoint.parse(
                    "dummy=tests.utils.dispatcher_examples.examples",
                    get_distribution("triad"),
                ),
                EntryPoint.parse(
                    "dummy2=tests.utils.dispatcher_examples.invalid",
                    get_distribution("triad"),
                )
            ]
        },
    )

    from tests.utils.dispatcher_examples import dtest

    assert 1 == dtest(1)
    assert 2 == dtest("ab")
    assert 2 == dtest(dict(a=1, b=2))
