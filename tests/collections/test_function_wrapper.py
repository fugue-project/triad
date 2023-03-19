from typing import Any, Callable, Dict, Iterable, List, Optional
from copy import deepcopy
import pandas as pd
from pytest import raises

from triad.collections.function_wrapper import (
    AnnotatedParam,
    FunctionWrapper,
    NoneParam,
    OtherParam,
    function_wrapper,
)
from triad.exceptions import InvalidOperationError
from triad import to_uuid


class _Dummy:
    pass


@function_wrapper(None)
class MockFunctionWrapper(FunctionWrapper):
    pass


@MockFunctionWrapper.annotated_param(pd.DataFrame, "d", child_can_reuse_code=True)
class DfParam(AnnotatedParam):
    pass


@MockFunctionWrapper.annotated_param(_Dummy)
class Df2Param(DfParam):
    pass


@MockFunctionWrapper.annotated_param("[Series]", "s", lambda a: a == pd.Series)
class SeriesParam(AnnotatedParam):
    pass


def test_registration():
    with raises(InvalidOperationError):

        @function_wrapper(None)
        class _F1:  # not a subclass of AnnotatedParam
            pass

    with raises(InvalidOperationError):

        @FunctionWrapper.annotated_param(_Dummy, ".")
        class _D1:  # not a subclass of AnnotatedParam
            pass

    with raises(InvalidOperationError):

        @FunctionWrapper.annotated_param(
            _Dummy
        )  # _NoneParam doesn't allow child to reuse the code
        class _D2(NoneParam):
            pass


def test_misc():
    f = deepcopy(FunctionWrapper(f2))
    assert f(1, 2, c=3, b=4) == 1 + 2 + 4 - 3

    f = deepcopy(FunctionWrapper(f1))
    assert f.input_code == "xx"
    assert f.output_code == "n"

    f = deepcopy(MockFunctionWrapper(f1))
    assert f.input_code == "ds"
    assert f.output_code == "n"

    f = MockFunctionWrapper(f6)
    assert isinstance(f._params["a"], Df2Param)


def test_uuid():
    x = FunctionWrapper(f2)
    y = deepcopy(x)
    z = FunctionWrapper(f5)
    assert to_uuid(x) == to_uuid(y)
    assert to_uuid(y) != to_uuid(z)


def test_parse_annotation():
    p = FunctionWrapper.parse_annotation(None)
    assert p.code == "x"
    assert isinstance(p, OtherParam)
    p = FunctionWrapper.parse_annotation(type(None))
    assert p.code == "x"
    assert isinstance(p, OtherParam)
    p = FunctionWrapper.parse_annotation(type(None), none_as_other=False)
    assert p.code == "n"
    assert isinstance(p, NoneParam)

    p = MockFunctionWrapper.parse_annotation(pd.DataFrame)
    assert p.code == "d"

    p = MockFunctionWrapper.parse_annotation(pd.Series)
    assert p.code == "s"


def test_parse_function():
    def _parse_function(f, params_re, return_re):
        MockFunctionWrapper(f, params_re, return_re)

    _parse_function(f1, "^ds$", "n")
    raises(TypeError, lambda: _parse_function(f1, "^xx$", "n"))
    _parse_function(f2, "^xxxx$", "n")
    _parse_function(f3, "^yz$", "n")
    _parse_function(f4, "^0x$", "d")
    _parse_function(f6, "^d$", "n")
    _parse_function(f7, "^yz$", "n")


def f1(a: pd.DataFrame, b: pd.Series) -> None:
    pass


def f2(e: int, a, b: int, c):
    return e + a + b - c


def f3(*args, **kwargs):
    pass


def f4(self, a) -> pd.DataFrame:
    pass


def f5(e: int, a, b: int, c):
    return e + a + b - c


def f6(a: _Dummy) -> None:
    pass


def f7(*args: Any, **kwargs: int):
    pass
