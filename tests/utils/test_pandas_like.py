import json
import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import pyarrow as pa
from pytest import raises
from triad.utils.pandas_like import PD_UTILS, _DEFAULT_DATETIME
from triad.utils.pyarrow import expression_to_schema
import pickle


def test_to_schema():
    df = pd.DataFrame([[1.0, 2], [2.0, 3]])
    raises(ValueError, lambda: PD_UTILS.to_schema(df))
    df = pd.DataFrame([[1.0, 2], [2.0, 3]], columns=["x", "y"])
    assert list(pa.Schema.from_pandas(df)) == list(PD_UTILS.to_schema(df))
    df = pd.DataFrame([["a", 2], ["b", 3]], columns=["x", "y"])
    assert list(pa.Schema.from_pandas(df)) == list(PD_UTILS.to_schema(df))
    df = pd.DataFrame([], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype("object")})
    assert [pa.field("x", pa.int32()), pa.field("y", pa.string())] == list(
        PD_UTILS.to_schema(df)
    )
    df = pd.DataFrame([[1, "x"], [2, "y"]], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype("object")})
    assert list(pa.Schema.from_pandas(df)) == list(PD_UTILS.to_schema(df))
    df = pd.DataFrame([[1, "x"], [2, "y"]], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype(str)})
    assert list(pa.Schema.from_pandas(df)) == list(PD_UTILS.to_schema(df))
    df = pd.DataFrame([[1, "x"], [2, "y"]], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype("str")})
    assert list(pa.Schema.from_pandas(df)) == list(PD_UTILS.to_schema(df))

    # timestamp test
    df = pd.DataFrame(
        [[datetime(2020, 1, 1, 2, 3, 4, 5), date(2020, 2, 2)]], columns=["a", "b"]
    )
    assert list(expression_to_schema("a:datetime,b:date")) == list(
        PD_UTILS.to_schema(df)
    )

    # test index
    df = pd.DataFrame([[3.0, 2], [2.0, 3]], columns=["x", "y"])
    df = df.sort_values(["x"])
    assert list(pa.Schema.from_pandas(df, preserve_index=False)) == list(
        PD_UTILS.to_schema(df)
    )
    df.index.name = "x"
    raises(ValueError, lambda: PD_UTILS.to_schema(df))
    df = df.reset_index(drop=True)
    assert list(pa.Schema.from_pandas(df)) == list(PD_UTILS.to_schema(df))
    df["p"] = "p"
    df = df.set_index(["p"])
    df.index.name = None
    raises(ValueError, lambda: PD_UTILS.to_schema(df))


def test_as_array_iterable():
    df = DF([], "a:str,b:int")
    assert [] == df.as_array()
    assert [] == df.as_array(type_safe=True)

    df = DF([["a", 1]], "a:str,b:int")
    assert [["a", 1]] == df.as_array()
    assert [["a", 1]] == df.as_array(["a", "b"])
    assert [[1, "a"]] == df.as_array(["b", "a"])
    assert [[1, "a"]] == df.as_array(["b", "a"], null_schema=True)

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
    assert [[datetime(2020, 1, 1), 1]] == df.as_array()
    assert isinstance(df.as_array(type_safe=True)[0][0], datetime)
    assert isinstance(df.as_array(type_safe=True)[0][1], int)

    df = DF([[pd.NaT, 1.1]], "a:datetime,b:int")
    assert df.as_array(type_safe=True)[0][0] is None
    assert isinstance(df.as_array(type_safe=True)[0][1], int)

    df = DF([[1.0, 1.1]], "a:double,b:int")
    assert [[1.0, 1]] == df.as_array(type_safe=True)
    assert isinstance(df.as_array()[0][0], float)
    assert isinstance(df.as_array()[0][1], int)


def test_as_array_iterable_datetime():
    df = pd.DataFrame(
        [[datetime(2020, 1, 1, 2, 3, 4, 5), date(2020, 2, 2)]], columns=["a", "b"]
    )
    v1 = list(PD_UTILS.as_array_iterable(df, type_safe=True))[0]
    v2 = list(
        PD_UTILS.as_array_iterable(
            df, schema=expression_to_schema("a:datetime,b:date"), type_safe=True
        )
    )[0]
    assert v1[0] == v2[0]
    assert not isinstance(v1[0], pd.Timestamp)
    assert type(v1[0]) == datetime
    assert type(v1[0]) == type(v2[0])

    assert v1[1] == v2[1]
    assert not isinstance(v1[1], pd.Timestamp)
    assert type(v1[1]) == date
    assert type(v1[1]) == type(v2[1])


def test_nested():
    # data = [[dict(b=[30, "40"])]]
    # s = expression_to_schema("a:{a:str,b:[int]}")
    # df = DF(data, "a:{a:str,b:[int]}")
    # a = df.as_array(type_safe=True)
    # assert [[dict(a=None, b=[30, 40])]] == a

    data = [[[json.dumps(dict(b=[30, "40"]))]]]
    s = expression_to_schema("a:[{a:str,b:[int]}]")
    df = DF(data, "a:[{a:str,b:[int]}]")
    a = df.as_array(type_safe=True)
    assert [[[dict(a=None, b=[30, 40])]]] == a

    data = [[json.dumps(["1", 2])]]
    s = expression_to_schema("a:[int]")
    df = DF(data, "a:[int]")
    a = df.as_array(type_safe=True)
    assert [[[1, 2]]] == a


def test_binary():
    b = pickle.dumps("xyz")
    data = [[b, b"xy"]]
    s = expression_to_schema("a:bytes,b:bytes")
    df = DF(data, "a:bytes,b:bytes")
    a = df.as_array(type_safe=True)
    assert [[b, b"xy"]] == a


def test_nan_none():
    df = DF([[None, None]], "b:str,c:double", True)
    assert df.native.iloc[0, 0] is None
    arr = df.as_array(type_safe=True)[0]
    assert arr[0] is None
    assert arr[1] is None

    df = DF([[None, None]], "b:int,c:bool", True)
    arr = df.as_array(type_safe=True)[0]
    assert arr[0] is None
    assert arr[1] is None

    df = DF([], "b:str,c:double", True)
    assert len(df.as_array()) == 0


def test_boolean_enforce():
    df = DF([[1, True], [2, False], [3, None]], "b:int,c:bool", True)
    arr = df.as_array(type_safe=True)
    assert [[1, True], [2, False], [3, None]] == arr

    df = DF([[1, 1], [2, 0]], "b:int,c:bool", True)
    arr = df.as_array(type_safe=True)
    assert [[1, True], [2, False]] == arr

    df = DF([[1, "trUe"], [2, "False"], [3, None]], "b:int,c:bool", True)
    arr = df.as_array(type_safe=True)
    assert [[1, True], [2, False], [3, None]] == arr


def test_fillna_default():
    df = pd.DataFrame([["a"], [None]], columns=["x"])
    s = PD_UTILS.fillna_default(df["x"])
    assert ["a", 0] == s.tolist()

    df = pd.DataFrame([["a"], ["b"]], columns=["x"])
    s = PD_UTILS.fillna_default(df["x"].astype(np.str_))
    assert ["a", "b"] == s.tolist()

    dt = datetime.now()
    df = pd.DataFrame([[dt], [None]], columns=["x"])
    s = PD_UTILS.fillna_default(df["x"])
    assert [dt, _DEFAULT_DATETIME] == s.tolist()

    df = pd.DataFrame([[True], [None]], columns=["x"])
    s = PD_UTILS.fillna_default(df["x"])
    assert [True, 0] == s.tolist()

    df = pd.DataFrame([[True], [False]], columns=["x"])
    s = PD_UTILS.fillna_default(df["x"].astype(bool))
    assert [True, False] == s.tolist()


def test_safe_group_by_apply():
    df = DF([["a", 1], ["a", 2], [None, 3]], "b:str,c:long", True)

    def _m1(df):
        PD_UTILS.ensure_compatible(df)
        df["ct"] = df.shape[0]
        return df

    res = PD_UTILS.safe_groupby_apply(df.native, ["b"], _m1)
    PD_UTILS.ensure_compatible(res)
    assert 3 == res.shape[0]
    assert 3 == res.shape[1]
    assert [["a", 1, 2], ["a", 2, 2], [None, 3, 1]] == res.values.tolist()

    res = PD_UTILS.safe_groupby_apply(df.native, [], _m1)
    PD_UTILS.ensure_compatible(res)
    assert 3 == res.shape[0]
    assert 3 == res.shape[1]
    assert [["a", 1, 3], ["a", 2, 3], [None, 3, 3]] == res.values.tolist()

    df = DF([[1.0, "a"], [1.0, "b"], [None, "c"], [None, "d"]], "b:double,c:str", True)
    res = PD_UTILS.safe_groupby_apply(df.native, ["b"], _m1)
    assert [
        [1.0, "a", 2],
        [1.0, "b", 2],
        [float("nan"), "c", 2],
        [float("nan"), "d", 2],
    ].__repr__() == res.values.tolist().__repr__()


def test_safe_group_by_apply_special_types():
    def _m1(df):
        PD_UTILS.ensure_compatible(df)
        df["ct"] = df.shape[0]
        return df

    df = DF(
        [["a", 1.0], [None, 3.0], [None, 3.0], [None, None]], "a:str,b:double", True
    )
    res = PD_UTILS.safe_groupby_apply(df.native, ["a", "b"], _m1)
    PD_UTILS.ensure_compatible(res)
    assert 4 == res.shape[0]
    assert 3 == res.shape[1]
    DF(
        [["a", 1.0, 1], [None, 3.0, 2], [None, 3.0, 2], [None, None, 1]],
        "a:str,b:double,ct:int",
        True,
    ).assert_eq(res)

    dt = datetime.now()
    df = DF([["a", dt], [None, dt], [None, dt], [None, None]], "a:str,b:datetime", True)
    res = PD_UTILS.safe_groupby_apply(df.native, ["a", "b"], _m1)
    PD_UTILS.ensure_compatible(res)
    assert 4 == res.shape[0]
    assert 3 == res.shape[1]
    DF(
        [["a", dt, 1], [None, dt, 2], [None, dt, 2], [None, None, 1]],
        "a:str,b:datetime,ct:int",
        True,
    ).assert_eq(res)

    dt = date(2020, 1, 1)
    df = DF([["a", dt], [None, dt], [None, dt], [None, None]], "a:str,b:date", True)
    res = PD_UTILS.safe_groupby_apply(df.native, ["a", "b"], _m1)
    PD_UTILS.ensure_compatible(res)
    assert 4 == res.shape[0]
    assert 3 == res.shape[1]
    DF(
        [["a", dt, 1], [None, dt, 2], [None, dt, 2], [None, None, 1]],
        "a:str,b:date,ct:int",
        True,
    ).assert_eq(res)

    dt = date(2020, 1, 1)
    df = DF([["a", dt], ["b", dt], ["b", dt], ["b", None]], "a:str,b:date", True)
    res = PD_UTILS.safe_groupby_apply(df.native, ["a", "b"], _m1)
    PD_UTILS.ensure_compatible(res)
    assert 4 == res.shape[0]
    assert 3 == res.shape[1]
    DF(
        [["a", dt, 1], ["b", dt, 2], ["b", dt, 2], ["b", None, 1]],
        "a:str,b:date,ct:int",
        True,
    ).assert_eq(res)


def test_is_compatile_index():
    df = DF([["a", 1], [None, 2]], "a:str,b:int", True)
    assert PD_UTILS.is_compatile_index(df.native)
    tdf = df.native.sort_values("a")
    assert PD_UTILS.is_compatile_index(tdf)
    tdf = tdf.set_index("a")
    assert not PD_UTILS.is_compatile_index(tdf)


class DF(object):  # This is a mock
    def __init__(self, data, schema, enforce=False):
        s = expression_to_schema(schema)
        df = pd.DataFrame(data, columns=s.names)
        self.native = PD_UTILS.enforce_type(df, s, enforce)
        self.schema = s

    def as_array(self, cols=None, type_safe=False, null_schema=False):
        schema = None if null_schema else self.schema
        return list(
            PD_UTILS.as_array_iterable(
                self.native, schema=schema, columns=cols, type_safe=type_safe
            )
        )

    def assert_eq(self, df):
        pd.testing.assert_frame_equal(self.native, df, check_dtype=False)
