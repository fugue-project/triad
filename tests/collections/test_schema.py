from triad.collections.schema import Schema
from collections import OrderedDict

def test_schema():
    s = Schema("a:int,b:str")
    assert s == "a:int,b:str"
    assert s == ["a:int", "b:str"]
    assert s == [("a", "int"), ("b", "str")]
    assert s == OrderedDict([("a", "int"), ("b", "str")])
