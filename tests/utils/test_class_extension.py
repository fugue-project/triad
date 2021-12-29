from triad.utils.class_extension import (
    extension_method,
    extensible_class,
    _get_first_arg_type,
)
from pytest import raises
from typing import Any, Optional


class AA:
    pass


def test_get_first_arg_type():
    def f1():
        pass

    # no arg
    raises(ValueError, lambda: _get_first_arg_type(f1))

    def f2(a):
        pass

    # no annotation
    raises(ValueError, lambda: _get_first_arg_type(f2))

    def f3(a, b: int):
        pass

    # no annotation
    raises(ValueError, lambda: _get_first_arg_type(f3))

    def f4(*args: int):
        pass

    # first arg must be keyword only
    raises(ValueError, lambda: _get_first_arg_type(f4))

    def f5(**kwargs: int):
        pass

    # first arg must be keyword only
    raises(ValueError, lambda: _get_first_arg_type(f5))

    def f6(x: Any):
        pass

    # first arg must be a valid type
    raises(ValueError, lambda: _get_first_arg_type(f6))

    def f7(x: Optional[AA]):
        pass

    # first arg must be a valid type
    raises(ValueError, lambda: _get_first_arg_type(f7))

    def f8(x: int):
        pass

    assert _get_first_arg_type(f8) == int

    def f9(x: AA):
        pass

    assert _get_first_arg_type(f9) == AA

    def f10(x: "AA"):
        pass

    assert _get_first_arg_type(f10) == AA


def test_class_extension():
    @extensible_class
    class A:
        def a(self):
            """aaa"""
            return 1

        def __getattr__(self, name: str) -> None:
            raise AttributeError(name)

    @extension_method(class_type=A)
    def em(a):
        """aaaem"""
        return a.a() * 2

    @extension_method(name="em1")
    def xx(a: A):
        """aaaem"""
        return a.a() * 2

    @extension_method
    def em2(a: A, b):
        """aaaem2"""
        return a.a() * 3 + b

    assert 2 == A().em()
    assert 2 == A().em1()
    assert 7 == A().em2(4)

    assert "aaa" == A.a.__doc__
    assert "aaaem" == A.em.__doc__
    assert "aaaem2" == A.em2.__doc__

    with raises(ValueError):  # conflicts with built-in

        @extension_method
        def a(a):
            """aaaem"""
            return a.a() * 2

    with raises(ValueError):  # conflicts with registered

        @extension_method(name="em")
        def xz(a: A):
            """aaaem"""
            return a.a() * 2

    @extension_method(on_dup="ignore", name="em")
    def xy(a: A):
        """aaaem"""
        return a.a() * 2
