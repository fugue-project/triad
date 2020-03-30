from triad.utils.assertion import assert_or_throw
from pytest import raises


def test_assert_or_throw():
    assert_or_throw(True)
    raises(AssertionError, lambda: assert_or_throw(False))
    raises(AssertionError, lambda: assert_or_throw(False, "test"))
    raises(AssertionError, lambda: assert_or_throw(False, 123))
    raises(TypeError, lambda: assert_or_throw(False, TypeError()))
    
