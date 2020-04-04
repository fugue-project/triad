from collections import OrderedDict
from datetime import datetime

import pyarrow as pa
from pytest import raises
from triad.collections.schema import Schema, SchemaError
from triad.exceptions import InvalidOperationError, NoneArgumentError
import pandas as pd


def test_schema_init():
    s = Schema("a:int,b:str")
    assert 2 == len(s)
    assert Schema(s) == "a:int,b:str"
    assert Schema(s) is not s
    assert Schema("a:int,b:str") == "a:int,b:str"
    assert Schema("a:int", "b:str") == "a:int,b:str"
    assert Schema(dict(a=int, b="str")) == "a:long,b:str"
    assert Schema(("a", int), [("b", str), ("c", "int")],
                  dict(d=datetime, e="str")) == "a:long,b:str,c:int,d:datetime,e:str"
    assert Schema(" a:{x:int32, y:str},b:[datetime]") == "a:{x:int,y:str},b:[datetime]"
    pa_schema = pa.schema([pa.field("123", pa.int32()), pa.field(
        "b", pa.string())])
    raises(SchemaError, lambda: Schema(pa_schema))
    assert 0 == len(Schema())
    assert 0 == len(Schema([]))
    assert 0 == len(Schema(None))
    assert 0 == len(Schema(""))


def test_schema_properties():
    s = Schema("a:int,b:str")
    assert ["a", "b"] == s.names
    assert [pa.int32(), pa.string()] == s.types
    assert [pa.field("a", pa.int32()), pa.field("b", pa.string())] == s.fields
    assert pa.schema([pa.field("a", pa.int32()), pa.field(
        "b", pa.string())]) == s.pyarrow_schema
    assert s.pyarrow_schema == s.pyarrow_schema


def test_schema_copy():
    a = Schema("a:int,b:str").copy()
    assert isinstance(a, Schema)
    assert a == "a:int,b:str"


def test_schema_setter():
    a = Schema("a:int,b:str")
    with raises(NoneArgumentError):
        a["c"] = None  # None is invalid
    with raises(SchemaError):
        a["b"] = "str"  # Update is not allowed
    with raises(SchemaError):
        a["123"] = "int"  # Col name is invalid
    with raises(SchemaError):
        a["x"] = pa.field("y", pa.int32())  # key!=field.name
    with raises(SchemaError):
        a["y"] = pa.binary()  # unsupported types
    a["c"] = str
    a["d"] = pa.field("d", pa.int32())
    assert a == "a:int,b:str,c:str,d:int"


def test_schema_eq():
    s = Schema("a:int,b:str")
    assert s != None
    assert not(s == None)
    assert s == s
    assert s == Schema("a:int,b:str")
    assert not(s == Schema("b:str,a:int"))
    assert s == ["a:int", "b:str"]
    assert s != ["a:long", "b:str"]
    assert not(s == ["a:long", "b:str"])
    assert s == [("a", "int"), ("b", str)]
    assert s == OrderedDict([("a", "int"), ("b", str)])


def test_schema_contains():
    s = Schema("a:int,b:str")
    assert None not in s
    assert s in s
    assert "a" in s
    assert "c" not in s
    assert "a:int" in s
    assert "a:long" not in s
    assert pa.field("a", pa.int32()) in s
    assert pa.field("aa", pa.int32()) not in s
    assert pa.field("a", pa.int64()) not in s
    assert ["a", ("b", str)] in s
    assert ["a", ("b", int)] not in s


def test_schema_append():
    s = Schema()
    s.append(pa.field("a", pa.int32()))
    assert s == "a:int"
    raises(SchemaError, lambda: s.append("b"))
    raises(SchemaError, lambda: s.append(123))
    s.append("b:str")
    assert s == "a:int,b:str"
    s.append(Schema("c:int").pa_schema)
    assert s == "a:int,b:str,c:int"
    s.append("")
    assert s == "a:int,b:str,c:int"
    s.append(" ")
    assert s == "a:int,b:str,c:int"
    df = pd.DataFrame([["a", 1], ["b", 2]], columns=["x", "y"])
    assert Schema(df) == "x:str,y:long"


def test_schema_remove():
    s = Schema("a:int,b:str,c:int")
    t = s.remove(s)
    assert t == ""
    t = s.remove(None)
    assert t == "a:int,b:str,c:int"
    assert t is not s
    t = s.remove("")
    assert t == "a:int,b:str,c:int"
    t = s.remove(" ")
    assert t == "a:int,b:str,c:int"
    t = s.remove("b")
    assert t == "a:int,c:int"
    t = s.remove("b:str")
    assert t == "a:int,c:int"
    t = s.remove("b:str", require_type_match=False)
    assert t == "a:int,c:int"
    t = s.remove("b:int", require_type_match=False)
    assert t == "a:int,c:int"
    t = s.remove("b:str", require_type_match=True)
    assert t == "a:int,c:int"
    raises(SchemaError, lambda: s.remove("b:int", require_type_match=True))
    t = s.remove("b:int", require_type_match=True, ignore_type_mismatch=True)
    assert t == "a:int,b:str,c:int"
    assert t is not s
    t = s.remove(Schema("c:int,b:str"))
    assert t == "a:int"
    t = s.remove(pa.field("c", pa.int32()))
    assert t == "a:int,b:str"
    t = s.remove(["c", "b"])
    assert t == "a:int"
    t = s.remove({"c", "b"})
    assert t == "a:int"
    t = s.remove(["c", ("b", str)])
    assert t == "a:int"
    t = s.remove(["c", [("b", str), "a"]])
    assert t == ""
    raises(SchemaError, lambda: s.remove("x"))
    t = s.remove("x", ignore_key_mismatch=True)
    assert t == s
    assert t is not s


def test_schema_extract():
    s = Schema("a:int,b:str,c:int")
    t = s.extract(s)
    assert t == s
    t = s.extract(None)
    assert t == ""
    t = s.extract("")
    assert t == ""
    t = s.extract(" ")
    assert t == ""
    t = s.extract("b")
    assert t == "b:str"
    t = s.extract("b:str")
    assert t == "b:str"
    t = s.extract("b:str", require_type_match=False)
    assert t == "b:str"
    t = s.extract("b:int", require_type_match=False)
    assert t == "b:str"
    t = s.extract("b:str", require_type_match=True)
    assert t == "b:str"
    raises(SchemaError, lambda: s.extract("b:int", require_type_match=True))
    t = s.extract("b:int", require_type_match=True, ignore_type_mismatch=True)
    assert t == ""
    assert t is not s
    t = s.extract(Schema("c:int,b:str"))
    assert t == "c:int,b:str"
    t = s.extract(pa.field("c", pa.int32()))
    assert t == "c:int"
    t = s.extract(["c", "b"])
    assert t == "c:int,b:str"
    raises(SchemaError, lambda: s.extract({"c", "b"}))
    t = s.extract(["c", ("b", str)])
    assert t == "c:int,b:str"
    t = s.extract(["c", [("b", str), "a"]])
    assert t == "c:int,b:str,a:int"
    raises(SchemaError, lambda: s.extract(["x", "b", "a"]))
    t = s.extract(["x", "b", "a"], ignore_key_mismatch=True)
    assert t == "b:str,a:int"
    t = s.extract("x:int,b:str,a:int", ignore_key_mismatch=True)
    assert t == "b:str,a:int"
    raises(SchemaError, lambda: s.extract("x:int,b:str,a:int"))


def test_schema_update_delete():
    s = Schema("a:int,b:str,c:int")
    with raises(SchemaError):
        del s["a"]
    with raises(SchemaError):
        del s["x"]
    with raises(SchemaError):
        s["a"] = str
    raises(SchemaError, lambda: s.pop("a"))
    raises(SchemaError, lambda: s.popitem("a"))
    raises(SchemaError, lambda: s.update(dict(a=int)))


def test_schema_operators():
    s = Schema("a:int,b:str,c:int")
    s += "d:int"
    t = s + "e:int"
    t += ""
    assert s == "a:int,b:str,c:int,d:int"
    assert t == "a:int,b:str,c:int,d:int,e:int"
    t = s - ""
    assert t == s
    t = s - ["a", "c"]
    assert t == "b:str,d:int"
    with raises(SchemaError):
        t -= "a"
    assert t == "b:str,d:int"


def test_schema_set_ops():
    s = Schema("a:int,b:str,c:int")
    s1 = Schema("a:int,f:str")
    s2 = Schema("a:str,f:str")
    assert s.exclude(s1) == "b:str,c:int"
    raises(SchemaError, lambda: s.exclude(s2))
    assert s.exclude(s2, require_type_match=False) == "b:str,c:int"
    assert s.exclude(s2, ignore_type_mismatch=True) == "a:int,b:str,c:int"

    s3 = Schema("a:int,b:int")
    assert s.intersect(s3) == "a:int"
    assert s.intersect(s3, require_type_match=False) == "a:int,b:str"
    raises(SchemaError, lambda: s.intersect(s3, ignore_type_mismatch=False))
    s4 = Schema("b:int,a1:str,a:int,b1:str")
    assert s4.intersect(s3) == "b:int,a:int"
    assert s4.exclude(s3) == "a1:str,b1:str"
    assert s4.intersect(["b1", "b"], use_other_order=True) == "b1:str,b:int"

    s5 = Schema("e:str,c:int,b:int,d:int")
    assert s.union(s5) == "a:int,b:str,c:int,e:str,d:int"
    raises(SchemaError, lambda: s.union(s5, require_type_match=True))
    assert s.union("e:str") == "a:int,b:str,c:int,e:str"
    assert s == "a:int,b:str,c:int"
    s.union_with("e:str")
    assert s == "a:int,b:str,c:int,e:str"