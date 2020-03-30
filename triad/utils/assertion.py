from typing import Union


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
