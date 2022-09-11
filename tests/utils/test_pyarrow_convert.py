import math
from collections import OrderedDict
from datetime import date, datetime

import pandas as pd
import pyarrow as pa
import numpy as np
from pytest import raises
from triad.collections.schema import Schema
from triad.constants import FLOAT_INF, FLOAT_NAN, FLOAT_NINF
from triad.exceptions import InvalidOperationError, NoneArgumentError
from triad.utils.pyarrow import _to_pydate, _to_pydatetime, apply_schema
import pickle

"""
None,"1",1,1.1,"2020-01-01","2020-01-01 01:02:03",
pd.NaT,pd.Timestamp,float(nan),float(-inf),float(inf),
true,True,false,False

str,int,double,bool,datetime,date,decimal,dict,list

"""


def test_convert_to_null():
    pdt = pd.Timestamp("2020-01-01T02:03:04")

    _test_convert(None, "null", None)
    _test_convert("1", "null", None)
    _test_convert(pd.NaT, "null", None)
    _test_convert(pdt, "null", None)
    _test_convert(FLOAT_NAN, "null", None)
    _test_convert(FLOAT_INF, "null", None)
    _test_convert(FLOAT_NINF, "null", None)
    _test_convert(True, "null", None)
    _test_convert(False, "null", None)


def test_convert_to_str():
    pdt = pd.Timestamp("2020-01-01T02:03:04")

    _test_convert(None, "str", None)
    _test_convert("1", "str", "1")
    _test_convert(pd.NaT, "str", "NaT")
    _test_convert(pdt, "str", "2020-01-01 02:03:04")
    _test_convert(FLOAT_NAN, "str", "nan")
    _test_convert(FLOAT_INF, "str", "inf")
    _test_convert(FLOAT_NINF, "str", "-inf")
    _test_convert(True, "str", "True")
    _test_convert(False, "str", "False")


def test_convert_to_int():
    pdt = pd.Timestamp("2020-01-01T02:03:04")

    _test_convert(None, "int", None)
    _test_convert("1", "int", 1)
    _test_convert(1, "int", 1)
    _assert_raise("1.1", "int")
    _test_convert(1.1, "int", 1)
    _test_convert(pd.NaT, "int", None)
    _assert_raise(pdt, "int")
    _test_convert(FLOAT_NAN, "int", None)
    _test_convert(np.nan, "int", None)
    _assert_raise(FLOAT_INF, "int")
    _assert_raise(FLOAT_NINF, "int")
    _assert_raise("true", "int")


def test_convert_to_double():
    pdt = pd.Timestamp("2020-01-01T02:03:04")

    _test_convert(None, "double", None)
    _test_convert("1", "double", 1.0)
    _test_convert(1.1, "double", 1.1)
    _test_convert("1.1", "double", 1.1)
    _test_convert(pd.NaT, "double", None)
    _assert_raise(pdt, "double")
    _test_convert(FLOAT_NAN, "double", None)
    _test_convert("nan", "double", None)
    _test_convert("NaN", "double", None)
    _test_convert(FLOAT_INF, "double", FLOAT_INF)
    _test_convert("inf", "double", FLOAT_INF)
    _test_convert("INF", "double", FLOAT_INF)
    _test_convert(FLOAT_NINF, "double", FLOAT_NINF)
    _test_convert("-inf", "double", FLOAT_NINF)
    _test_convert("-INF", "double", FLOAT_NINF)
    _assert_raise("true", "double")


def test_convert_to_decimal():
    _assert_not_supported(None, "decimal(5,2)")


def test_convert_to_bool():
    pdt = pd.Timestamp("2020-01-01T02:03:04")

    _test_convert(None, "bool", None)
    _test_convert(True, "bool", True)
    _test_convert(False, "bool", False)
    _test_convert("true", "bool", True)
    _test_convert("True", "bool", True)
    _test_convert("false", "bool", False)
    _test_convert("False", "bool", False)
    _test_convert(pd.NaT, "bool", None)
    _assert_raise(pdt, "bool")
    _test_convert(FLOAT_NAN, "bool", None)
    _test_convert(np.nan, "bool", None)


def test_convert_to_binary():
    pdt = pd.Timestamp("2020-01-01T02:03:04")

    _test_convert(None, "bytes", None)
    _test_convert(b"\x0e\x15", "bytes", b"\x0e\x15")
    _test_convert(bytearray(b"\x0e\x15"), "bytes", b"\x0e\x15")
    _test_convert(False, "bytes", pickle.dumps(False))
    _test_convert("true", "bytes", pickle.dumps("true"))
    _test_convert(pd.NaT, "bytes", pickle.dumps(pd.NaT))
    _test_convert(pdt, "bytes", pickle.dumps(pdt))
    _test_convert(FLOAT_NAN, "bytes", pickle.dumps(FLOAT_NAN))
    _test_convert(np.nan, "bytes", pickle.dumps(np.nan))


def test_convert_to_datetime():
    pdt = pd.Timestamp("2020-01-01T02:03:04")
    dt = datetime(2020, 1, 1, 2, 3, 4)
    d = date(2020, 1, 1)
    _test_convert(None, "datetime", None)
    _assert_raise("1", "datetime")
    _test_convert("2020-01-01 02:03:04", "datetime", dt)
    _test_convert("2020-01-01", "datetime", datetime(2020, 1, 1))
    _test_convert(pd.NaT, "datetime", None)
    _test_convert(pdt, "datetime", dt)
    assert isinstance(_to_pydatetime(pdt), datetime)
    assert not isinstance(_to_pydatetime(pdt), pd.Timestamp)
    _test_convert(dt, "datetime", dt)
    _test_convert(d, "datetime", datetime(2020, 1, 1))
    _assert_raise(FLOAT_NAN, "datetime")


def test_convert_to_date():
    pdt = pd.Timestamp("2020-01-01T02:03:04")
    dt = datetime(2020, 1, 1, 2, 3, 4)
    d = date(2020, 1, 1)
    _test_convert(None, "date", None)
    _assert_raise("1", "date")
    _test_convert("2020-01-01 02:03:04", "date", d)
    _test_convert("2020-01-01", "date", d)
    _test_convert(pd.NaT, "date", None)
    _test_convert(pdt, "date", d)
    assert isinstance(_to_pydate(pdt), date)
    assert not isinstance(_to_pydate(pdt), pd.Timestamp)
    _test_convert(dt, "date", d)
    _test_convert(d, "date", d)
    _assert_raise(FLOAT_NAN, "date")


def test_convert_to_dict_shallow():
    d = dict(a=1)
    _test_convert(None, "{a:int}", None)
    _test_convert(d, "{a:int}", d)
    _test_convert({}, "{a:int}", {})
    _assert_raise("1", "{a:int}")
    _assert_raise("abc", "{a:int}")
    _assert_raise(pd.NaT, "{a:int}")
    _assert_raise(FLOAT_NAN, "{a:int}")
    _assert_raise(True, "{a:int}")


def test_convert_to_dict_deep():
    d = dict(a=1)
    _test_convert_nested(None, "{a:int}", None)
    _test_convert_nested(d, "{a:int}", d)
    _test_convert_nested({}, "{a:int}", dict(a=None))
    _test_convert_nested(
        '{"b":{"c":["1"]}}', "{a:int,b:{c:[int]}}", dict(a=None, b=dict(c=[1]))
    )
    _assert_raise(dict(a="x"), "{a:int}", True)
    _assert_raise(dict(b=123), "{b:{a:int}}", True)


def test_convert_to_list_shallow():
    d = [1]
    _test_convert(None, "[int]", None)
    _test_convert(d, "[int]", d)
    _test_convert([], "[int]", [])
    _assert_raise("1", "[int]")
    _assert_raise("abc", "[int]")
    _assert_raise(pd.NaT, "[int]")
    _assert_raise(FLOAT_NAN, "[int]")
    _assert_raise(True, "[int]")
    _assert_raise(np.ndarray([1, 2]), "[int]", [1, 2])


def test_convert_to_list_deep():
    d = ["1"]
    _test_convert_nested(None, "[int]", None)
    _test_convert_nested(d, "[int]", [1])
    _test_convert_nested([], "[int]", [])
    _test_convert_nested(
        '[{"b":{"c":["1"]}}]', "[{a:int,b:{c:[int]}}]", [dict(a=None, b=dict(c=[1]))]
    )
    _test_convert_nested([d], "[[int]]", [[1]])
    _assert_raise(["x"], "[int]", True)
    _assert_raise(1, "[int]", True)


def test_convert_to_map_shallow():
    _test_convert(None, "<str,int>", None)
    _test_convert({"a": 1}, "<str,int>", {"a": 1})
    _test_convert([("a", 1)], "<str,int>", [("a", 1)])
    _test_convert({}, "<str,int>", {})
    _assert_raise("1", "<str,int>")
    _assert_raise("abc", "<str,int>")
    _assert_raise(pd.NaT, "<str,int>")
    _assert_raise(FLOAT_NAN, "<str,int>")
    _assert_raise(True, "<str,int>")
    _assert_raise(np.ndarray([1, 2]), "<str,int>")


def test_convert_to_map_deep():
    _test_convert_nested(None, "<str,int>", None)
    _test_convert_nested({"a": "1", "b": 1}, "<str,int>", [("a", 1), ("b", 1)])
    _test_convert_nested([("a", "1")], "<str,int>", [("a", 1)])
    _test_convert_nested('{"a":{"b":"1"}}', "<str,<str,int>>", [("a", [("b", 1)])])
    _assert_raise({"a": "x"}, "<str,int>", True)
    _assert_raise(1, "<str,int>", True)


def _test_convert(orig, expected_type, expected_value):
    a = [[orig]]
    s = Schema("a:" + expected_type).pa_schema
    x = list(apply_schema(s, a))[0]
    y = list(apply_schema(s, a, copy=False))[0]
    for b in [x, y]:
        if isinstance(expected_value, float) and math.isnan(expected_value):
            assert math.isnan(b[0])
        elif expected_value is pd.NaT:
            assert b[0] is pd.NaT
        else:
            assert expected_value == b[0]
    assert x is not a[0]
    assert y is a[0]


def _test_convert_nested(orig, expected_type, expected_value):
    a = [[orig]]
    s = Schema("a:" + expected_type).pa_schema
    x = list(apply_schema(s, a, deep=True))[0]
    y = list(apply_schema(s, a, copy=False, deep=True))[0]
    for b in [x, y]:
        assert expected_value == b[0]
    assert x is not a[0]
    assert y is a[0]


def _assert_raise(orig, expected_type, deep=False):
    with raises(ValueError):
        a = [[orig]]
        s = Schema("a:" + expected_type).pa_schema
        b = list(apply_schema(s, a, deep=deep))[0]


def _assert_not_supported(orig, expected_type, deep=False):
    with raises(NotImplementedError):
        a = [[orig]]
        s = Schema("a:" + expected_type).pa_schema
        b = list(apply_schema(s, a, deep=deep))[0]
