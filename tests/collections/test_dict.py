import json
import pickle
from copy import copy, deepcopy
from typing import Any

from pytest import raises
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError


def test_indexed_orderd_dict():
    d = IndexedOrderedDict([("b", 2), ("a", 1)])
    d1 = IndexedOrderedDict([("a", 1), ("b", 2)])
    assert dict(a=1, b=2) == d
    assert d1 != d
    assert d._need_reindex
    assert 1 == d.index_of_key("a")
    assert not d._need_reindex
    assert "a" == d.get_key_by_index(1)
    assert 2 == d.get_value_by_index(0)
    assert ("a", 1) == d.get_item_by_index(1)
    assert not d._need_reindex
    d.set_value_by_index(1, 10)
    assert not d._need_reindex
    assert ("a", 10) == d.get_item_by_index(1)
    assert ("b", 2) == d.pop_by_index(0)
    assert d._need_reindex
    assert 1 == len(d)
    assert 0 == d.index_of_key("a")
    assert not d._need_reindex
    assert 10 == d.setdefault("a", 20)
    assert not d._need_reindex
    assert 30 == d.setdefault("b", 30)
    assert d._need_reindex
    d.clear()
    assert d._need_reindex
    raises(KeyError, lambda: d.index_of_key("a"))
    assert not d._need_reindex
    assert 0 == len(d)

    d = IndexedOrderedDict([("b", 2), ("a", 1)])
    assert not d.readonly
    d.set_readonly()
    assert d.readonly
    raises(InvalidOperationError, lambda: d.__setitem__("b", "3"))
    raises(InvalidOperationError, lambda: d.__delitem__("b"))
    assert 2 == d["b"]

    # popitem
    d = IndexedOrderedDict([("b", 2), ("a", 1), ("c", 3)])
    assert 1 == d.index_of_key("a")
    assert not d._need_reindex
    assert ("b", 2) == d.popitem(last=False)
    assert d._need_reindex
    assert ("c", 3) == d.popitem(last=True)
    assert 0 == d.index_of_key("a")
    assert not d._need_reindex
    d.set_readonly()
    raises(InvalidOperationError, lambda: d.popitem())

    # move_to_end
    d = IndexedOrderedDict([("b", 2), ("a", 1), ("c", 3)])
    d1 = IndexedOrderedDict([("b", 2), ("c", 3), ("a", 1)])
    assert d != d1
    d.move_to_end("a")
    assert d == d1
    d.set_readonly()
    raises(InvalidOperationError, lambda: d.move_to_end("b"))

    # copy and deepcopy
    d = IndexedOrderedDict([("b", 2), ("a", 1), ("c", 3)])
    d.set_readonly()
    d.index_of_key("a")
    assert not d._need_reindex
    d1 = d.copy()
    assert isinstance(d1, IndexedOrderedDict)
    assert not d1._need_reindex
    assert d == d1
    assert 1 == d1.index_of_key("a")
    assert not d1.readonly  # after copy, readonly is set to False
    del d1["a"]  # will not affect the original
    assert 1 == d.index_of_key("a")

    d = IndexedOrderedDict([("b", [1, IndexedOrderedDict([("x", [2, 4])]), 3])])
    d.set_readonly()
    d1 = copy(d)
    assert not d1.readonly  # after copy, readonly is set to False
    d1["b"][0] = 10
    assert 10 == d["b"][0]
    d1["b"][1]["x"][0] = 200
    assert 200 == d["b"][1]["x"][0]
    d.index_of_key("b")
    assert not d._need_reindex
    d2 = deepcopy(d)
    assert d2._need_reindex  # after deepcopy, reindex is required
    assert not d2.readonly  # after deepcopy, readonly is set to False
    d2["b"][0] = 20
    assert 10 == d["b"][0]
    d2["b"][1]["x"][0] = 300
    assert 200 == d["b"][1]["x"][0]

    # pickle
    d = IndexedOrderedDict([("b", 2), ("a", 1), ("c", 3)])
    d.set_readonly()
    d.index_of_key("a")
    assert not d._need_reindex
    d1 = pickle.loads(pickle.dumps(d))
    assert isinstance(d1, IndexedOrderedDict)
    assert not d1._need_reindex
    assert d == d1
    assert 1 == d1.index_of_key("a")
    assert d1.readonly

    # equals
    d = IndexedOrderedDict([("b", 2), ("a", 1), ("c", 3)])
    d.set_readonly()
    d1 = IndexedOrderedDict([("b", 2), ("c", 3), ("a", 1)])
    d2 = [("b", 2), ("a", 1), ("c", 3)]
    d3 = [("b", 2), ("c", 3), ("a", 1)]
    d4 = dict([("b", 2), ("c", 3), ("a", 1)])
    assert not d.equals(d1, True)
    assert d.equals(d1, False)
    assert d.equals(d2, True)
    assert d.equals(d2, False)
    assert not d.equals(d3, True)
    assert d.equals(d3, False)
    assert not d.equals(d4, True)
    assert d.equals(d4, False)


def test_param_dict():
    d = ParamDict([("a", 1), ("b", 2)])
    assert 1 == d["a"]
    assert 1 == d[0]
    assert 2 == d["b"]
    assert "2" == d.get_or_throw(1, str)
    # if giving index, it should ignore the throw flag and always throw
    raises(IndexError, lambda: d.get(2, "x"))
    raises(IndexError, lambda: d.get_or_none(2, str))

    d = {"a": "b", "b": {"x": 1, "y": "d"}}
    p = ParamDict(d)
    print({"test": p})
    d["b"]["x"] = 2
    assert 1 == p["b"]["x"]
    p = ParamDict(d, deep=False)
    d["b"]["x"] = 3
    assert 3 == p["b"]["x"]
    pp = ParamDict(p, deep=False)
    p["b"]["x"] = 4
    assert 4 == pp["b"]["x"]
    pp = ParamDict(p, deep=True)
    p["b"]["x"] = 5
    assert 4 == pp["b"]["x"]

    assert 2 == len(p)
    assert "a,b" == ",".join([k for k, _ in p.items()])
    del p["a"]
    assert 1 == len(p)
    p["c"] = 1
    assert 2 == len(p)
    assert "c" in p
    assert "a" not in p

    raises(ValueError, lambda: p.get("c", None))
    assert 1 == p.get("c", 2)
    assert "1" == p.get("c", "2")
    assert 1.0 == p.get("c", 2.0)
    raises(TypeError, lambda: p.get("c", ParamDict()))
    assert 2 == p.get("d", 2)
    p["arr"] = [1, 2]
    assert [1, 2] == p.get("arr", [])
    assert [] == p.get("arr2", [])

    assert p.get_or_none("e", int) is None
    assert 1 == p.get_or_none("c", int)
    assert "1" == p.get_or_none("c", str)
    # exists but can't convert type
    raises(TypeError, lambda: p.get_or_none("c", ParamDict))

    raises(KeyError, lambda: p.get_or_throw("e", int))
    assert 1 == p.get_or_throw("c", int)
    assert "1" == p.get_or_throw("c", str)
    # exists but can't convert type
    raises(TypeError, lambda: p.get_or_throw("c", ParamDict))

    p = ParamDict()
    assert 0 == len(p)
    for x in p:
        pass

    raises(TypeError, lambda: ParamDict("abc"))

    a = ParamDict({"a": 1, "b": 2})
    b = ParamDict({"b": 2, "a": 1})
    c = ParamDict({"b": 2})
    assert a == a
    assert a != b
    assert a != c
    assert a == {"b": 2, "a": 1}
    assert a != {"b": 1, "a": 1}
    assert a != None
    assert not (a == None)

    p = ParamDict(
        {
            "a": "True",
            "b": True,
            "c": "true",
            "d": "False",
            "e": False,
            "f": "false",
            "g": "yes",
            "h": "NO",
            "i": 0,
            "j": "1",
            "k": "",
        }
    )
    assert p.get_or_throw("a", bool)
    assert p.get_or_throw("b", bool)
    assert p.get_or_throw("c", bool)
    assert not p.get_or_throw("d", bool)
    assert not p.get_or_throw("e", bool)
    assert not p.get_or_throw("f", bool)
    assert p.get_or_throw("g", bool)
    assert not p.get_or_throw("h", bool)
    assert not p.get_or_throw("i", bool)
    assert p.get_or_throw("j", bool)
    raises(TypeError, lambda: p.get_or_throw("k", bool))

    s = '{"a":false,"b":10,"c":"cd"}'
    p = ParamDict(json.loads(s))
    assert not p.get_or_throw("a", bool)
    assert "10" == p.get_or_throw("b", str)
    assert "cd" == p.get_or_throw("c", str)
    raises(KeyError, lambda: p.get_or_throw("d", str))

    print(p.to_json())
    print(p.to_json(True))

    # update
    p = ParamDict(dict(a=1, b=2))
    p1 = ParamDict(dict(b=3, c=4))
    p.update(p1)
    assert dict(a=1, b=3, c=4) == p

    p = ParamDict(dict(a=1, b=2))
    p.update(p1, ParamDict.OVERWRITE)
    assert dict(a=1, b=3, c=4) == p

    p = ParamDict(dict(a=1, b=2))
    p.update(p1, ParamDict.IGNORE)
    assert dict(a=1, b=2, c=4) == p

    p = ParamDict(dict(a=1, b=2))
    raises(KeyError, lambda: p.update(p1, ParamDict.THROW))

    raises(ValueError, lambda: p.update(p1, 100))

    p.set_readonly()
    raises(InvalidOperationError, lambda: p.update(p1, 100))


def test_using_indexed_ordered_dict():
    def get_count(d: IndexedOrderedDict[str, int]):
        return len(d)

    dd = IndexedOrderedDict(dict(a=1))
    assert 1 == get_count(dd)
