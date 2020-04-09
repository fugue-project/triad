from triad.utils.json import loads_no_dup
from pytest import raises


def test_loads_no_dup():
    assert dict(x=1, y=2) == loads_no_dup('{"x": 1, "y": 2}')
    raises(KeyError, lambda: loads_no_dup('{"x": 1, "x": 2}'))
