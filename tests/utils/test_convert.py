from triad.utils.convert import _parse_value_and_unit, to_size
import numpy as np
from pytest import raises


def test_to_size():
    raises(ValueError, lambda: to_size(None))
    raises(ValueError, lambda: to_size(""))
    raises(ValueError, lambda: to_size("abc"))
    raises(AssertionError, lambda: to_size("-1"))
    raises(AssertionError, lambda: to_size(-1))
    raises(ValueError, lambda: to_size("1xx"))
    assert 0 == to_size(0)
    assert 1 == to_size(1)
    assert 1 == to_size(1.9)
    assert 10 == to_size(" 1 0 B ")
    assert 10 * 1024 == to_size(" 10k")
    assert 10 * 1024 * 1024 == to_size(" 10 m b")
    assert 10 * 1024 * 1024 * 1024 == to_size("10g")
    assert 10 * 1024 * 1024 * 1024 * 1024 == to_size("10tb")
    assert int(1.1 * 1024 * 1024) == to_size(" 1 . 1 mb ")


def test_parse_value_and_unit():
    raises(ValueError, lambda: _parse_value_and_unit(None))
    raises(ValueError, lambda: _parse_value_and_unit(""))
    raises(ValueError, lambda: _parse_value_and_unit("abc"))
    assert (1.0, "") == _parse_value_and_unit(1)
    assert (1.1, "") == _parse_value_and_unit(1.1)
    assert (1.1, "") == _parse_value_and_unit(1.1)
    assert (1.1, "") == _parse_value_and_unit(np.float32(1.1))
    assert (1.0, "") == _parse_value_and_unit(" 1 ")
    assert (-1.0, "") == _parse_value_and_unit(" -1.0 ")
    assert (-1.0, "m10") == _parse_value_and_unit(" - 1 . 0 m 1 0 ")
