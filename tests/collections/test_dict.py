import pickle
from copy import deepcopy, copy

from pytest import raises
from triad.collections.dict import IndexedOrderedDict


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

    # popitem
    d = IndexedOrderedDict([("b", 2), ("a", 1), ("c", 3)])
    assert 1 == d.index_of_key("a")
    assert not d._need_reindex
    assert ("b", 2) == d.popitem(last=False)
    assert d._need_reindex
    assert ("c", 3) == d.popitem(last=True)
    assert 0 == d.index_of_key("a")
    assert not d._need_reindex

    # move_to_end
    d = IndexedOrderedDict([("b", 2), ("a", 1), ("c", 3)])
    d1 = IndexedOrderedDict([("b", 2), ("c", 3), ("a", 1)])
    assert d != d1
    d.move_to_end("a")
    assert d == d1

    # copy and deepcopy
    d = IndexedOrderedDict([("b", 2), ("a", 1), ("c", 3)])
    d.index_of_key("a")
    assert not d._need_reindex
    d1 = d.copy()
    assert isinstance(d1, IndexedOrderedDict)
    assert not d1._need_reindex
    assert d == d1
    assert 1 == d1.index_of_key("a")
    del d1["a"]  # will not affect the original
    assert 1 == d.index_of_key("a")

    d = IndexedOrderedDict([("b", [1, IndexedOrderedDict([("x", [2, 4])]), 3])])
    d1 = copy(d)
    d1["b"][0] = 10
    assert 10 == d["b"][0]
    d1["b"][1]["x"][0]=200
    assert 200 == d["b"][1]["x"][0]
    d2 = deepcopy(d)
    d2["b"][0] = 20
    assert 10 == d["b"][0]
    d2["b"][1]["x"][0]=300
    assert 200 == d["b"][1]["x"][0]

    # pickle
    d = IndexedOrderedDict([("b", 2), ("a", 1), ("c", 3)])
    d.index_of_key("a")
    assert not d._need_reindex
    d1 = pickle.loads(pickle.dumps(d))
    assert isinstance(d1, IndexedOrderedDict)
    assert not d1._need_reindex
    assert d == d1
    assert 1 == d1.index_of_key("a")
