from typing import Any
from triad.exceptions import NoneArgumentError


def assert_or_throw(bool_exp: bool, exception: Any = None) -> None:
    """Assert on expression and throw custom exception

    :param bool_exp: boolean expression to assert on
    :param exception: a custom Exception instance, or any other object that
        will be stringfied and instantiate an AssertionError, or a function
        that can generate the supported data types

    .. admonition:: Examples

        .. code-block:: python

            assert_or_throw(True, "assertion error")
            assert_or_throw(False)  # raise AssertionError
            assert_or_throw(False, "assertion error")  # raise AssertionError
            assert_or_throw(False, TypeError("assertion error"))  # raise TypeError

            # Lazy evaluations is useful when constructing the error
            # itself is expensive or error-prone. With lazy evaluations, happy
            # path will be fast and error free.
            def fail():  # a function that is slow and wrong
                sleep(10)
                raise TypeError

            assert_or_throw(True, fail())  # (unexpectedly) raise TypeError
            assert_or_throw(True, fail)  # no exception
            assert_or_throw(True, lambda: "a" + fail())  # no exception
            assert_or_throw(False, lambda: "a" + fail())  # raise TypeError

    """
    if not bool_exp:
        _exception: Any = exception
        if callable(exception):
            _exception = exception()
        if _exception is None:
            raise AssertionError()
        if isinstance(_exception, Exception):
            raise _exception
        if isinstance(_exception, str):
            raise AssertionError(_exception)
        raise AssertionError(str(_exception))


def assert_arg_not_none(obj: Any, arg_name: str = "", msg: str = "") -> None:
    """Assert an argument is not None, otherwise raise exception

    :param obj: argument value
    :param arg_name: argument name, if None or empty, it will use `msg`
    :param msg: only when `arg_name` is None or empty, this value is used

    :raises NoneArgumentError: with `arg_name` or `msg`
    """
    if obj is None:
        if arg_name != "" and arg_name is not None:
            msg = f"{arg_name} can't be None"
        msg = msg or ""
        raise NoneArgumentError(msg)
