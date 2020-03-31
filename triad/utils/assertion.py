from typing import Union, Any
from triad.exceptions import NoneArgumentError


def assert_or_throw(
    bool_exp: bool, exception: Union[None, str, Exception] = None
) -> None:
    """Assert on expression and throw custom exception

    :param bool_exp: boolean expression to assert on
    :param exception: a custom Exception instance, or any other object that
        will be stringfied and instantiate an AssertionError
    """
    if not bool_exp:
        if isinstance(exception, Exception):
            raise exception
        if isinstance(exception, str):
            raise AssertionError(exception)
        if exception is None:
            raise AssertionError()
        raise AssertionError(str(exception))


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
