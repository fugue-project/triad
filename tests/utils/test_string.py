from triad.utils.string import assert_triad_var_name
from pytest import raises


def test_assert_triad_var_name():
    raises(AssertionError, lambda: assert_triad_var_name(None))
    raises(AssertionError, lambda: assert_triad_var_name(""))
    raises(AssertionError, lambda: assert_triad_var_name(" "))
    raises(AssertionError, lambda: assert_triad_var_name("1"))
    raises(AssertionError, lambda: assert_triad_var_name(123))
    raises(AssertionError, lambda: assert_triad_var_name("_"))
    raises(AssertionError, lambda: assert_triad_var_name("__"))
    raises(AssertionError, lambda: assert_triad_var_name("a "))
    raises(AssertionError, lambda: assert_triad_var_name("a a"))
    raises(AssertionError, lambda: assert_triad_var_name("大大"))
    raises(AssertionError, lambda: assert_triad_var_name("ŋ"))
    assert "a" == assert_triad_var_name("a")
    assert "_1" == assert_triad_var_name("_1")
    assert "_a_1" == assert_triad_var_name("_a_1")
