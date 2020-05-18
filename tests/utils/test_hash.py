from __future__ import absolute_import

from triad.utils.hash import to_uuid


def test_to_uuid():
    assert to_uuid() == to_uuid()
    assert to_uuid(None) == to_uuid(None)
    assert to_uuid() != to_uuid(None)

    d = {"a": 1, "b": {"x": "x", "y": [1, 2, 3]}}
    id1 = to_uuid(d)
    id2 = to_uuid(d)
    d["b"]["y"][0] = None
    id3 = to_uuid(d)
    d["a"] = "1"
    id4 = to_uuid(d)
    assert id1 == id2
    assert id1 != id3
    assert id4 != id3

    id1 = to_uuid([0, 1])
    id2 = to_uuid([1, 0])
    id3 = to_uuid(x for x in [0, 1])
    assert id1 != id2
    assert id1 == id3

    id1 = to_uuid(["a", "aa"])
    id2 = to_uuid(["aa", "a"])
    assert id1 != id2

    assert to_uuid([Mock2(), Mock2()]) == to_uuid([Mock(), Mock()])
    assert to_uuid([Mock2(2), Mock2(3)]) == to_uuid([Mock(2), Mock(3)])
    assert to_uuid([Mock2(2), Mock2(3)]) != to_uuid([Mock(3), Mock(3)])


class Mock(object):
    def __init__(self, n=1):
        self.n = n

    def __uuid__(self) -> str:
        return str(self.n)


class Mock2(object):
    def __init__(self, n=1):
        self.n = n

    def __uuid__(self) -> int:
        return self.n
