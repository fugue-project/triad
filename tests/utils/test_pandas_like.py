import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
from pytest import raises
from triad.utils.pandas_like import (_ensure_compatible_index,
                                     as_array_iterable, enforce_type,
                                     safe_groupby_apply, to_schema)
from triad.utils.pyarrow import expression_to_schema


def test_to_schema():
    df = pd.DataFrame([[1.0, 2], [2.0, 3]])
    raises(ValueError, lambda: to_schema(df))
    df = pd.DataFrame([[1.0, 2], [2.0, 3]], columns=["x", "y"])
    assert list(pa.Schema.from_pandas(df)) == list(to_schema(df))
    df = pd.DataFrame([["a", 2], ["b", 3]], columns=["x", "y"])
    assert list(pa.Schema.from_pandas(df)) == list(to_schema(df))
    df = pd.DataFrame([], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype('object')})
    assert [pa.field("x", pa.int32()), pa.field(
        "y", pa.string())] == list(to_schema(df))
    df = pd.DataFrame([[1, "x"], [2, "y"]], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype('object')})
    assert list(pa.Schema.from_pandas(df)) == list(to_schema(df))
    df = pd.DataFrame([[1, "x"], [2, "y"]], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype(str)})
    assert list(pa.Schema.from_pandas(df)) == list(to_schema(df))
    df = pd.DataFrame([[1, "x"], [2, "y"]], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype('str')})
    assert list(pa.Schema.from_pandas(df)) == list(to_schema(df))

    # test index
    df = pd.DataFrame([[3.0, 2], [2.0, 3]], columns=["x", "y"])
    df = df.sort_values(["x"])
    assert list(pa.Schema.from_pandas(df)) == list(to_schema(df))
    df.index.name = "x"
    raises(ValueError, lambda: to_schema(df))
    df = df.reset_index(drop=True)
    assert list(pa.Schema.from_pandas(df)) == list(to_schema(df))
    df["p"] = "p"
    df = df.set_index(["p"])
    df.index.name = None
    raises(ValueError, lambda: to_schema(df))


def test_as_array_iterable():
    df = DF([], "a:str,b:int")
    assert [] == df.as_array()
    assert [] == df.as_array(type_safe=True)

    df = DF([["a", 1]], "a:str,b:int")
    assert [["a", 1]] == df.as_array()
    assert [["a", 1]] == df.as_array(["a", "b"])
    assert [[1, "a"]] == df.as_array(["b", "a"])

    # prevent pandas auto type casting
    df = DF([[1.0, 1.1]], "a:double,b:int")
    assert [[1.0, 1]] == df.as_array()
    assert isinstance(df.as_array()[0][0], float)
    assert isinstance(df.as_array()[0][1], int)
    assert [[1.0, 1]] == df.as_array(["a", "b"])
    assert [[1, 1.0]] == df.as_array(["b", "a"])

    df = DF([[np.float64(1.0), 1.1]], "a:double,b:int")
    assert [[1.0, 1]] == df.as_array()
    assert isinstance(df.as_array()[0][0], float)
    assert isinstance(df.as_array()[0][1], int)

    df = DF([[pd.Timestamp("2020-01-01"), 1.1]], "a:datetime,b:int")
    df.native["a"] = pd.to_datetime(df.native["a"])
    assert [[datetime(2020, 1, 1), 1]] == df.as_array()
    assert isinstance(df.as_array()[0][0], datetime)
    assert isinstance(df.as_array()[0][1], int)

    df = DF([[pd.NaT, 1.1]], "a:datetime,b:int")
    df.native["a"] = pd.to_datetime(df.native["a"])
    assert isinstance(df.as_array()[0][0], datetime)
    assert isinstance(df.as_array()[0][1], int)

    df = DF([[1.0, 1.1]], "a:double,b:int")
    assert [[1.0, 1]] == df.as_array(type_safe=True)
    assert isinstance(df.as_array()[0][0], float)
    assert isinstance(df.as_array()[0][1], int)


def test_nested():
    data = [[[json.dumps(dict(b=[30, "40"]))]]]
    s = expression_to_schema("a:[{a:str,b:[int]}]")
    df = DF(data, "a:[{a:str,b:[int]}]")
    a = df.as_array(s, type_safe=True)
    assert [[[dict(a=None, b=[30, 40])]]] == a

    data = [[json.dumps(["1", 2])]]
    s = expression_to_schema("a:[int]")
    df = DF(data, "a:[int]")
    a = df.as_array(s, type_safe=True)
    assert [[[1, 2]]] == a


def test_nan_none():
    df = DF([[None, None]], "b:str,c:double", True)
    assert df.native.iloc[0, 0] is None
    arr = df.as_array(null_safe=True)[0]
    assert arr[0] is None
    assert math.isnan(arr[1])

    df = DF([[None, None]], "b:int,c:bool", True)
    arr = df.as_array(type_safe=True)[0]
    assert np.isnan(arr[0])  # TODO: this will cause inconsistent behavior cross engine
    assert np.isnan(arr[1])  # TODO: this will cause inconsistent behavior cross engine

    df = DF([["a", 1.1], [None, None]], "b:str,c:double", True)
    arr = df.as_array()[1]
    assert arr[0] is None
    assert math.isnan(arr[1])


def test_safe_group_by_apply():
    df = DF([["a", 1], ["a", 2], [None, 3]], "b:str,c:long", True)

    def _m1(df):
        _ensure_compatible_index(df)
        df["ct"] = df.shape[0]
        return df

    res = safe_groupby_apply(df.native, ["b"], _m1)
    _ensure_compatible_index(res)
    assert 3 == res.shape[0]
    assert 3 == res.shape[1]
    assert [["a", 1, 2], ["a", 2, 2], [None, 3, 1]] == res.values.tolist()


class DF(object):  # This is a mock
    def __init__(self, data, schema, enforce=False):
        s = expression_to_schema(schema)
        df = pd.DataFrame(data, columns=s.names)
        self.native = enforce_type(df, s, enforce)

    def as_array(self, cols=None, type_safe=False, null_safe=False):
        if cols is None or isinstance(cols, pa.Schema):
            return list(as_array_iterable(
                self.native, schema=cols, type_safe=type_safe, null_safe=null_safe))
        if isinstance(cols, list):
            os = to_schema(self.native)
            s = pa.schema([os.field(x) for x in cols])
            return list(as_array_iterable(
                self.native, schema=s, type_safe=type_safe, null_safe=null_safe))
