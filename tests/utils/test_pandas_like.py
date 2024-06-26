import json
import pickle
from datetime import date, datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pytest import raises

from triad.utils.pandas_like import _DEFAULT_DATETIME, PD_UTILS
from triad.utils.pyarrow import (
    PYARROW_VERSION,
    expression_to_schema,
    pa_table_to_pandas,
)

from .._utils import assert_df_eq


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

    df = pd.Series([], name="col", dtype="string").to_frame()
    assert expression_to_schema("col:str") == PD_UTILS.to_schema(df)

    df = pd.Series([], name="col", dtype="datetime64[ns]").to_frame()
    assert expression_to_schema("col:datetime") == PD_UTILS.to_schema(df)

    df = pd.Series([], name="col", dtype="datetime64[ns, UTC]").to_frame()
    assert expression_to_schema("col:timestamp(us, UTC)") == PD_UTILS.to_schema(df)

    # timestamp test
    df = pd.DataFrame(
        [[datetime(2020, 1, 1, 2, 3, 4, 5), date(2020, 2, 2)]], columns=["a", "b"]
    )
    assert list(expression_to_schema("a:datetime,b:date")) == list(
        PD_UTILS.to_schema(df)
    )

    # large string test (pandas>=2.2)
    df = pd.DataFrame([["abc"]], columns=["a"])
    df = df.astype("string[pyarrow]")
    assert PD_UTILS.to_schema(df) == expression_to_schema("a:str")

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
    np.issubdtype(df.as_array()[0][0], np.float64)
    np.issubdtype(df.as_array()[0][1], np.integer)
    assert [[1.0, 1]] == df.as_array(["a", "b"])
    assert [[1, 1.0]] == df.as_array(["b", "a"])

    df = DF([[np.float64(1.0), 1.1]], "a:double,b:int")
    assert [[1.0, 1]] == df.as_array()
    np.issubdtype(df.as_array()[0][0], np.float64)
    np.issubdtype(df.as_array()[0][1], np.integer)

    df = DF([[pd.Timestamp("2020-01-01"), 1.1]], "a:datetime,b:int")
    assert [[datetime(2020, 1, 1), 1]] == df.as_array()
    assert isinstance(df.as_array(type_safe=True)[0][0], datetime)
    assert isinstance(df.as_array(type_safe=True)[0][1], int)

    df = DF([[pd.NaT, 1.1]], "a:datetime,b:int")
    assert df.as_array(type_safe=True)[0][0] is None
    assert isinstance(df.as_array(type_safe=True)[0][1], int)

    df = DF([[1.0, 1.1]], "a:double,b:int")
    assert [[1.0, 1]] == df.as_array(type_safe=True)
    np.issubdtype(df.as_array()[0][0], np.float64)
    np.issubdtype(df.as_array()[0][1], np.integer)


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


@pytest.mark.skipif(PYARROW_VERSION.major < 8, reason="pyarrow>=8.0.0 required")
def test_nested():
    data = [[dict(b=[30, 40]), 2, 2.2]]
    df = DF(data, "a:{b:[int]},b:int,c:double")
    res = df.as_array(type_safe=True)
    assert res == data
    assert isinstance(res[0][1], int)
    assert not isinstance(res[0][1], np.number)
    assert isinstance(res[0][2], float)
    assert not isinstance(res[0][2], np.number)


def test_binary():
    b = pickle.dumps("xyz")
    data = [[b, b"xy"]]
    s = expression_to_schema("a:bytes,b:bytes")
    df = DF(data, "a:bytes,b:bytes")
    a = df.as_array(type_safe=True)
    assert [[b, b"xy"]] == a


def test_nan_none():
    df = DF([[None, None]], "b:str,c:double", True)
    assert pd.isna(df.native.iloc[0, 0])
    arr = df.as_array(type_safe=True)[0]
    assert arr[0] is None
    assert arr[1] is None

    df = DF([[None, None]], "b:int,c:bool", True)
    arr = df.as_array(type_safe=True)[0]
    assert arr[0] is None
    assert arr[1] is None

    df = DF([], "b:str,c:double", True)
    assert len(df.as_array()) == 0


def test_cast_df():
    df = pd.DataFrame(
        dict(
            a=pd.Series([1, 2], dtype="int32"), b=pd.Series([True, False], dtype="bool")
        )
    )
    assert df is PD_UTILS.cast_df(
        df,
        expression_to_schema("a:int32,b:bool"),
        use_extension_types=False,
        use_arrow_dtype=False,
    )
    res = PD_UTILS.cast_df(
        df,
        expression_to_schema("a:long,b:str"),
        use_extension_types=True,
        use_arrow_dtype=False,
    )
    assert isinstance(res.a.dtype, pd.Int64Dtype)
    assert isinstance(res.b.dtype, pd.StringDtype)
    assert res.values.tolist() == [[1, "true"], [2, "false"]]

    df = pd.DataFrame(dict(a=pd.Series(dtype="int32"), b=pd.Series(dtype="bool")))
    res = PD_UTILS.cast_df(
        df,
        expression_to_schema("a:long,b:str"),
        use_extension_types=True,
        use_arrow_dtype=False,
    )
    assert isinstance(res.a.dtype, pd.Int64Dtype)
    assert isinstance(res.b.dtype, pd.StringDtype)
    assert len(res) == 0


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


def test_timestamp_enforce():
    df = DF(
        [
            [1, pd.Timestamp("2020-01-02 00:00:00", tz="UTC")],
            [2, pd.Timestamp("2020-01-03 00:00:00", tz="UTC")],
        ],
        "b:int,c:datetime",
        True,
    )
    arr = df.as_array(type_safe=True)
    assert [
        [1, pd.Timestamp("2020-01-02 00:00:00")],
        [2, pd.Timestamp("2020-01-03 00:00:00")],
    ] == arr

    df = DF(
        [
            [1, pd.Timestamp("2020-01-02 00:00:00")],
            [2, pd.Timestamp("2020-01-03 00:00:00")],
        ],
        "b:int,c:timestamp(ns, UTC)",
        True,
    )
    arr = df.as_array(type_safe=True)
    assert [
        [1, pd.Timestamp("2020-01-02 00:00:00", tz="UTC")],
        [2, pd.Timestamp("2020-01-03 00:00:00", tz="UTC")],
    ] == arr

    df = DF(
        [
            [1, pd.Timestamp("2020-01-02 00:00:00", tz="US/Pacific")],
            [2, pd.Timestamp("2020-01-03 00:00:00", tz="US/Pacific")],
        ],
        "b:int,c:timestamp(ns, UTC)",
        True,
    )
    arr = df.as_array(type_safe=True)
    assert [
        [1, pd.Timestamp("2020-01-02 08:00:00", tz="UTC")],
        [2, pd.Timestamp("2020-01-03 08:00:00", tz="UTC")],
    ] == arr

    df = DF(
        [
            [1, "2020-01-02 00:00:00"],
            [2, "2020-01-03 00:00:00"],
        ],
        "b:int,c:datetime",
        True,
    )
    arr = df.as_array(type_safe=True)
    assert [
        [1, pd.Timestamp("2020-01-02 00:00:00")],
        [2, pd.Timestamp("2020-01-03 00:00:00")],
    ] == arr

    df = DF(
        [
            [1, "2020-01-02 00:00:00-0500"],
            [2, "2020-01-03 00:00:00-0500"],
        ],
        "b:int,c:timestamp(ns, UTC)",
        True,
    )
    arr = df.as_array(type_safe=True)
    assert [
        [1, pd.Timestamp("2020-01-02 05:00:00", tz="UTC")],
        [2, pd.Timestamp("2020-01-03 05:00:00", tz="UTC")],
    ] == arr

    if hasattr(pd, "ArrowDtype"):
        df = pd.DataFrame(dict(a=pd.Series(["2020-01-02"], dtype="string[pyarrow]")))
        res = PD_UTILS.cast_df(df, expression_to_schema("a:datetime"))
        assert res.a.iloc[0] == pd.Timestamp("2020-01-02")


def test_fillna_default():
    df = pd.DataFrame([[1.0], [None]], columns=["x"])
    s = PD_UTILS.fillna_default(df["x"])
    assert [1.0, 0.0] == s.tolist()

    df = pd.DataFrame([["a"], [None]], columns=["x"])
    s = PD_UTILS.fillna_default(df["x"])
    assert ["a", ""] == s.tolist()

    df = pd.DataFrame([["a"], ["b"]], columns=["x"])
    s = PD_UTILS.fillna_default(df["x"].astype(np.str_))
    assert ["a", "b"] == s.tolist()

    dt = datetime.now()
    df = pd.DataFrame([[dt], [None]], columns=["x"])
    s = PD_UTILS.fillna_default(df["x"])
    assert [dt, _DEFAULT_DATETIME] == s.tolist()

    df = pd.DataFrame([[True], [None]], columns=["x"])
    s = PD_UTILS.fillna_default(df["x"])
    assert [True, ""] == s.tolist()

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
    assert [["a", 1, 2], ["a", 2, 2], [None, 3, 1]] == res.astype(object).where(
        ~pd.isna(res), None
    ).values.tolist()

    res = PD_UTILS.safe_groupby_apply(df.native, [], _m1)
    PD_UTILS.ensure_compatible(res)
    assert 3 == res.shape[0]
    assert 3 == res.shape[1]
    assert [["a", 1, 3], ["a", 2, 3], [None, 3, 3]] == res.astype(object).where(
        ~pd.isna(res), None
    ).values.tolist()

    df = DF([[1.0, "a"], [1.0, "b"], [None, "c"], [None, "d"]], "b:double,c:str", True)
    res = PD_UTILS.safe_groupby_apply(df.native, ["b"], _m1)
    assert [
        [1.0, "a", 2],
        [1.0, "b", 2],
        [pd.NA, "c", 2],
        [pd.NA, "d", 2],
    ].__repr__() == res.values.tolist().__repr__()


def test_to_parquet_friendly():
    if hasattr(pd, "ArrowDtype"):
        adf = pa.Table.from_pandas(pd.DataFrame(dict(a=["a", "b"], c=[1, 2])))
        pdf = pa_table_to_pandas(adf, use_extension_types=True, use_arrow_dtype=True)
        res = PD_UTILS.to_parquet_friendly(pdf)
        assert res is pdf

        adf = pa.Table.from_pandas(pd.DataFrame(dict(a=[["a", "b"], ["c"]], c=[1, 2])))
        pdf = pa_table_to_pandas(adf, use_extension_types=False, use_arrow_dtype=True)
        res = PD_UTILS.to_parquet_friendly(pdf)
        assert res.dtypes["a"] == np.dtype(object)
        assert res.dtypes["c"] == pd.ArrowDtype(pa.int64())
        pdf = pa_table_to_pandas(adf, use_extension_types=True, use_arrow_dtype=True)
        res = PD_UTILS.to_parquet_friendly(pdf)
        assert res.dtypes["a"] == np.dtype(object)
        assert res.dtypes["c"] == pd.Int64Dtype()
        res = PD_UTILS.to_parquet_friendly(pdf, partition_cols=["c"])
        assert res.dtypes["a"] == np.dtype(object)
        assert res.dtypes["c"] == np.dtype(object)


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


def test_is_compatible_index():
    df = DF([["a", 1], [None, 2]], "a:str,b:int", True)
    assert PD_UTILS.is_compatile_index(df.native)
    tdf = df.native.sort_values("a")
    assert PD_UTILS.is_compatile_index(tdf)
    tdf = tdf.set_index("a")
    assert not PD_UTILS.is_compatile_index(tdf)


def test_parse_join_types():
    assert "cross" == PD_UTILS.parse_join_type("CROss")
    assert "inner" == PD_UTILS.parse_join_type("join")
    assert "inner" == PD_UTILS.parse_join_type("Inner")
    assert "left_outer" == PD_UTILS.parse_join_type("left")
    assert "left_outer" == PD_UTILS.parse_join_type("left  outer")
    assert "right_outer" == PD_UTILS.parse_join_type("right")
    assert "right_outer" == PD_UTILS.parse_join_type("right_ outer")
    assert "full_outer" == PD_UTILS.parse_join_type("full")
    assert "full_outer" == PD_UTILS.parse_join_type(" outer ")
    assert "full_outer" == PD_UTILS.parse_join_type("full_outer")
    assert "left_anti" == PD_UTILS.parse_join_type("anti")
    assert "left_anti" == PD_UTILS.parse_join_type("left anti")
    assert "left_semi" == PD_UTILS.parse_join_type("semi")
    assert "left_semi" == PD_UTILS.parse_join_type("left semi")
    raises(
        NotImplementedError,
        lambda: PD_UTILS.parse_join_type("right semi"),
    )


def test_drop_duplicates():
    def assert_eq(df, expected, expected_cols):
        res = PD_UTILS.drop_duplicates(df)
        assert_df_eq(res, expected, expected_cols)

    a = _to_df([["x", "a"], ["x", "a"], [None, None]], ["a", "b"])
    assert_eq(a, [["x", "a"], [None, None]], ["a", "b"])


def test_union():
    def assert_eq(df1, df2, unique, expected, expected_cols):
        res = PD_UTILS.union(df1, df2, unique=unique)
        assert_df_eq(res, expected, expected_cols)

    a = _to_df([["x", "a"], ["x", "a"], [None, None]], ["a", "b"])
    b = _to_df([["xx", "aa"], [None, None], ["a", "x"]], ["b", "a"])
    assert_eq(
        a,
        b,
        False,
        [
            ["x", "a"],
            ["x", "a"],
            [None, None],
            ["xx", "aa"],
            [None, None],
            ["a", "x"],
        ],
        ["a", "b"],
    )
    assert_eq(
        a,
        b,
        True,
        [["x", "a"], ["xx", "aa"], [None, None], ["a", "x"]],
        ["a", "b"],
    )


def test_intersect():
    def assert_eq(df1, df2, unique, expected, expected_cols):
        res = PD_UTILS.intersect(df1, df2, unique=unique)
        assert_df_eq(res, expected, expected_cols)

    a = _to_df([["x", "a"], ["x", "a"], [None, None]], ["a", "b"])
    b = _to_df([["xx", "aa"], [None, None], [None, None], ["a", "x"]], ["b", "a"])
    assert_eq(a, b, False, [[None, None]], ["a", "b"])
    assert_eq(a, b, True, [[None, None]], ["a", "b"])
    b = _to_df([["xx", "aa"], [None, None], ["x", "a"]], ["b", "a"])
    assert_eq(a, b, False, [["x", "a"], ["x", "a"], [None, None]], ["a", "b"])
    assert_eq(a, b, True, [["x", "a"], [None, None]], ["a", "b"])


def test_except():
    def assert_eq(df1, df2, unique, expected, expected_cols):
        res = PD_UTILS.except_df(df1, df2, unique=unique)
        assert_df_eq(res, expected, expected_cols)

    a = _to_df([["x", "a"], ["x", "a"], [None, None]], ["a", "b"])
    b = _to_df([["xx", "aa"], [None, None], ["a", "x"]], ["b", "a"])
    assert_eq(a, b, False, [["x", "a"], ["x", "a"]], ["a", "b"])
    assert_eq(a, b, True, [["x", "a"]], ["a", "b"])
    b = _to_df([["xx", "aa"], [None, None], ["x", "a"]], ["b", "a"])
    assert_eq(a, b, False, [], ["a", "b"])
    assert_eq(a, b, True, [], ["a", "b"])


def test_joins():
    def assert_eq(df1, df2, join_type, on, expected, expected_cols):
        res = PD_UTILS.join(df1, df2, join_type=join_type, on=on)
        assert_df_eq(res, expected, expected_cols)

    df1 = _to_df([[0, 1], [2, 3]], ["a", "b"])
    df2 = _to_df([[0, 10], [20, 30]], ["a", "c"])
    df3 = _to_df([[0, 1], [None, 3]], ["a", "b"])
    df4 = _to_df([[0, 10], [None, 30]], ["a", "c"])
    assert_eq(df1, df2, "inner", ["a"], [[0, 1, 10]], ["a", "b", "c"])
    assert_eq(df3, df4, "inner", ["a"], [[0, 1, 10]], ["a", "b", "c"])
    assert_eq(df1, df2, "left_semi", ["a"], [[0, 1]], ["a", "b"])
    assert_eq(df3, df4, "left_semi", ["a"], [[0, 1]], ["a", "b"])
    assert_eq(df1, df2, "left_anti", ["a"], [[2, 3]], ["a", "b"])
    assert_eq(df3, df4, "left_anti", ["a"], [[None, 3]], ["a", "b"])
    assert_eq(
        df1,
        df2,
        "left_outer",
        ["a"],
        [[0, 1, 10], [2, 3, None]],
        ["a", "b", "c"],
    )
    assert_eq(
        df3,
        df4,
        "left_outer",
        ["a"],
        [[0, 1, 10], [None, 3, None]],
        ["a", "b", "c"],
    )
    assert_eq(
        df1,
        df2,
        "right_outer",
        ["a"],
        [[0, 1, 10], [20, None, 30]],
        ["a", "b", "c"],
    )
    assert_eq(
        df3,
        df4,
        "right_outer",
        ["a"],
        [[0, 1, 10], [None, None, 30]],
        ["a", "b", "c"],
    )
    assert_eq(
        df1,
        df2,
        "full_outer",
        ["a"],
        [[0, 1, 10], [2, 3, None], [20, None, 30]],
        ["a", "b", "c"],
    )
    assert_eq(
        df3,
        df4,
        "full_outer",
        ["a"],
        [[0, 1, 10], [None, 3, None], [None, None, 30]],
        ["a", "b", "c"],
    )

    df1 = _to_df([[0, 1], [None, 3]], ["a", "b"])
    df2 = _to_df([[0, 10], [None, 30]], ["c", "d"])
    assert_eq(
        df1,
        df2,
        "cross",
        [],
        [
            [0, 1, 0, 10],
            [None, 3, 0, 10],
            [0, 1, None, 30],
            [None, 3, None, 30],
        ],
        ["a", "b", "c", "d"],
    )


def _to_df(data, cols):
    return pd.DataFrame(data, columns=cols)


class DF:  # This is a mock
    def __init__(self, data, schema, enforce=False):
        s = expression_to_schema(schema)
        df = pd.DataFrame(data, columns=s.names)
        self.native = PD_UTILS.cast_df(
            df, s, use_extension_types=True, use_arrow_dtype=False
        )
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
