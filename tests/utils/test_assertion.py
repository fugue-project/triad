from triad.utils.assertion import assert_or_throw, assert_arg_not_none
from pytest import raises
from triad.exceptions import NoneArgumentError


def test_assert_or_throw():
    assert_or_throw(True)
    raises(AssertionError, lambda: assert_or_throw(False))
    raises(AssertionError, lambda: assert_or_throw(False, "test"))
    raises(AssertionError, lambda: assert_or_throw(False, 123))
    raises(TypeError, lambda: assert_or_throw(False, TypeError()))


def test_assert_arg_not_none():
    assert_arg_not_none(1)
    with raises(NoneArgumentError) as err:
        assert_arg_not_none(None, "a")
    assert "a can't be None" == err.value.args[0]
    with raises(NoneArgumentError) as err:
        assert_arg_not_none(None, "a", "b")
    assert "a can't be None" == err.value.args[0]
    with raises(NoneArgumentError) as err:
        assert_arg_not_none(None, "", msg="b")
    assert "b" == err.value.args[0]
    with raises(NoneArgumentError) as err:
        assert_arg_not_none(None, None, msg="b")
    assert "b" == err.value.args[0]
