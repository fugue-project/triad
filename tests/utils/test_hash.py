from __future__ import absolute_import

from triad.utils.hash import to_uuid


def test_to_uuid():
    assert to_uuid() == to_uuid()
    assert to_uuid(None) == to_uuid(None)
    assert to_uuid() != to_uuid(None)

    d = {
        "a": 1,
        "b": {
            "x": "x",
            "y": [1, 2, 3]
        }
    }
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
