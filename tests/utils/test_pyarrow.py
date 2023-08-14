import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from packaging import version
from pytest import raises

from triad.utils.pyarrow import (
    LARGE_TYPES_REPLACEMENT,
    PYARROW_VERSION,
    TRIAD_DEFAULT_TIMESTAMP,
    SchemaedDataPartitioner,
    _parse_type,
    _type_to_expression,
    cast_pa_array,
    cast_pa_table,
    expression_to_schema,
    get_alter_func,
    get_eq_func,
    is_supported,
    pa_table_to_pandas,
    replace_type,
    replace_types_in_schema,
    replace_types_in_table,
    schema_to_expression,
    schemas_equal,
    to_pa_datatype,
    to_pandas_dtype,
    to_single_pandas_dtype,
)


def test_version():
    assert PYARROW_VERSION.major > 5
    assert PYARROW_VERSION.major < 100


def test_expression_conversion():
    _assert_from_expr("a:int,b:ubyte")
    _assert_from_expr(" a : int32 , b : uint8 ", "a:int,b:ubyte")
    _assert_from_expr("a:[int32],b:uint8", "a:[int],b:ubyte")
    _assert_from_expr(
        "a : { x : int32 , y : [string] } , b : [ uint8 ] ",
        "a:{x:int,y:[str]},b:[ubyte]",
    )
    _assert_from_expr(
        "a : [{ x : int32 , y : [string] }] , b : [ uint8 ] ",
        "a:[{x:int,y:[str]}],b:[ubyte]",
    )
    _assert_from_expr("a : < str , int32 > , b : int", "a:<str,int>,b:int")
    _assert_from_expr("a : < str , [int32] > , b : int", "a:<str,[int]>,b:int")
    _assert_from_expr("a:decimal(5,2)")
    _assert_from_expr("a:bytes,b:bytes")
    _assert_from_expr("a:bytes,b: binary", "a:bytes,b:bytes")

    # special chars
    _assert_from_expr("`` :bytes,b:str", "``:bytes,b:str")
    _assert_from_expr("```` :bytes,b:str", "````:bytes,b:str")
    _assert_from_expr("`\\` :bytes,b:str", "`\\`:bytes,b:str")
    _assert_from_expr('`"` :bytes,b:str', '`"`:bytes,b:str')

    _assert_from_expr("`中国` :bytes,b:str", "`中国`:bytes,b:str")
    _assert_from_expr("`مثال` :bytes,b:str", "`مثال`:bytes,b:str")

    _assert_from_expr("`a` :bytes,b:str", "a:bytes,b:str")
    _assert_from_expr("`a b` :bytes,b:str", "`a b`:bytes,b:str")
    _assert_from_expr("`a``b` :bytes,b:str", "`a``b`:bytes,b:str")
    _assert_from_expr("123:bytes,b:str", "`123`:bytes,b:str")
    _assert_from_expr("_:bytes,b:str", "`_`:bytes,b:str")
    _assert_from_expr("`__`:bytes,b:str", "`__`:bytes,b:str")

    raises(SyntaxError, lambda: expression_to_schema("int"))
    raises(SyntaxError, lambda: expression_to_schema(":int"))
    raises(SyntaxError, lambda: expression_to_schema("a:int,:int"))
    raises(SyntaxError, lambda: expression_to_schema(":int,a:int"))
    raises(SyntaxError, lambda: expression_to_schema("a:dummytype"))
    raises(SyntaxError, lambda: expression_to_schema("a:int,a:str"))
    raises(SyntaxError, lambda: expression_to_schema("a:int,b:{x:int,x:str}"))
    raises(SyntaxError, lambda: expression_to_schema("a:[int,str]"))
    raises(SyntaxError, lambda: expression_to_schema("a:[]"))
    raises(SyntaxError, lambda: expression_to_schema("a:<>"))
    raises(SyntaxError, lambda: expression_to_schema("a:<int>"))
    raises(SyntaxError, lambda: expression_to_schema("a:<int,str,str>"))
    raises(SyntaxError, lambda: expression_to_schema("a:int,`b:str"))


def test__parse_type():
    assert pa.int32() == _parse_type(" int ")
    assert TRIAD_DEFAULT_TIMESTAMP == _parse_type(" datetime ")
    assert pa.timestamp("s", "America/New_York") == _parse_type(
        " timestamp ( s , America/New_York ) "
    )
    assert pa.timestamp("s") == _parse_type(" timestamp ( s ) ")
    assert _parse_type(" timestamp ( us ) ") == _parse_type(" datetime ")
    assert pa.decimal128(5, 2) == _parse_type(" decimal(5,2) ")
    assert pa.decimal128(5) == _parse_type(" decimal ( 5 )  ")


def test__type_to_expression():
    assert "int" == _type_to_expression(pa.int32())
    assert "datetime" == _type_to_expression(TRIAD_DEFAULT_TIMESTAMP)
    assert "timestamp(ns,America/New_York)" == _type_to_expression(
        pa.timestamp("ns", "America/New_York")
    )
    assert "datetime" == _type_to_expression(pa.timestamp("s"))
    assert "datetime" == _type_to_expression(pa.timestamp("ns"))
    assert "datetime" == _type_to_expression(pa.timestamp("ms"))
    assert "datetime" == _type_to_expression(pa.timestamp("us"))
    assert "decimal(5)" == _type_to_expression(pa.decimal128(5))
    assert "decimal(5,2)" == _type_to_expression(pa.decimal128(5, 2))
    assert "bytes" == _type_to_expression(pa.binary())
    assert "bytes" == _type_to_expression(pa.binary(-1))
    raises(NotImplementedError, lambda: _type_to_expression(pa.binary(0)))
    raises(NotImplementedError, lambda: _type_to_expression(pa.binary(-2)))
    raises(NotImplementedError, lambda: _type_to_expression(pa.binary(1)))
    raises(NotImplementedError, lambda: _type_to_expression(pa.large_binary()))


def test_to_pa_datatype():
    assert pa.int32() == to_pa_datatype(pa.int32())
    assert pa.int32() == to_pa_datatype("int")
    assert pa.int64() == to_pa_datatype(int)
    assert pa.float64() == to_pa_datatype(float)
    assert pa.string() == to_pa_datatype(str)
    assert pa.bool_() == to_pa_datatype(bool)
    assert pa.float64() == to_pa_datatype(np.float64)
    assert TRIAD_DEFAULT_TIMESTAMP == to_pa_datatype(datetime)
    assert pa.date32() == to_pa_datatype(date)
    assert pa.date32() == to_pa_datatype("date")
    assert pa.binary() == to_pa_datatype("bytes")
    assert pa.binary() == to_pa_datatype("binary")

    assert pa.int8() == to_pa_datatype(pd.Series([1]).astype("Int8").dtype)
    assert pa.int16() == to_pa_datatype(pd.Series([1]).astype("Int16").dtype)
    assert pa.int32() == to_pa_datatype(pd.Series([1]).astype("Int32").dtype)
    assert pa.int64() == to_pa_datatype(pd.Series([1]).astype("Int64").dtype)

    assert pa.uint8() == to_pa_datatype(pd.Series([1]).astype("UInt8").dtype)
    assert pa.uint16() == to_pa_datatype(pd.Series([1]).astype("UInt16").dtype)
    assert pa.uint32() == to_pa_datatype(pd.Series([1]).astype("UInt32").dtype)
    assert pa.uint64() == to_pa_datatype(pd.Series([1]).astype("UInt64").dtype)

    assert pa.string() == to_pa_datatype(pd.Series(["x"]).astype("string").dtype)

    assert pa.bool_() == to_pa_datatype(pd.Series([True]).astype("boolean").dtype)

    if pd.__version__ >= "2":
        assert pa.string() == to_pa_datatype(
            pd.Series(["x"]).astype("string[pyarrow]").dtype
        )
        assert pa.int64() == to_pa_datatype(pd.ArrowDtype(pa.int64()))
        assert pa.string() == to_pa_datatype(pd.ArrowDtype(pa.string()))

    raises(TypeError, lambda: to_pa_datatype(123))
    raises(TypeError, lambda: to_pa_datatype(None))


def test_to_single_pandas_dtype():
    assert np.bool_ == to_single_pandas_dtype(pa.bool_(), False)
    assert np.int16 == to_single_pandas_dtype(pa.int16(), False)
    assert np.uint32 == to_single_pandas_dtype(pa.uint32(), False)
    assert np.float32 == to_single_pandas_dtype(pa.float32(), False)
    assert np.dtype(str) == to_single_pandas_dtype(pa.string(), False)
    assert np.dtype("<M8[ns]") == to_single_pandas_dtype(pa.timestamp("ns"), False)

    assert pd.BooleanDtype() == to_single_pandas_dtype(pa.bool_(), True)
    assert pd.Int16Dtype() == to_single_pandas_dtype(pa.int16(), True)
    assert pd.Int16Dtype() == to_single_pandas_dtype(
        pa.int16(), True, use_arrow_dtype=True
    )
    assert pd.UInt32Dtype() == to_single_pandas_dtype(pa.uint32(), True)
    assert pd.Float32Dtype() == to_single_pandas_dtype(pa.float32(), True)
    assert pd.StringDtype() == to_single_pandas_dtype(pa.string(), True)
    assert np.dtype("<M8[ns]") == to_single_pandas_dtype(pa.timestamp("ns"), True)

    assert np.dtype("O") == to_single_pandas_dtype(pa.list_(pa.string()), False)
    assert np.dtype("O") == to_single_pandas_dtype(pa.list_(pa.string()), True)
    assert np.dtype("O") == to_single_pandas_dtype(
        pa.struct([pa.field("a", pa.int32())]), True
    )
    assert np.dtype("O") == to_single_pandas_dtype(
        pa.struct([pa.field("a", pa.int32())]), False
    )

    if hasattr(pd, "ArrowDtype"):
        assert pd.ArrowDtype(pa.int16()) == to_single_pandas_dtype(
            pa.int16(), False, use_arrow_dtype=True
        )
        assert pd.ArrowDtype(pa.timestamp("ns")) == to_single_pandas_dtype(
            pa.timestamp("ns"), True, use_arrow_dtype=True
        )
        assert pd.ArrowDtype(pa.list_(pa.string())) == to_single_pandas_dtype(
            pa.list_(pa.string()), False, use_arrow_dtype=True
        )
        assert pd.ArrowDtype(
            pa.struct([pa.field("a", pa.int32())])
        ) == to_single_pandas_dtype(
            pa.struct([pa.field("a", pa.int32())]), False, use_arrow_dtype=True
        )


def test_to_pandas_dtype():
    schema = expression_to_schema("a:bool,b:int,c:double,d:string,e:datetime")
    res = to_pandas_dtype(schema, False)
    assert np.bool_ == res["a"]
    assert np.int32 == res["b"]
    assert np.float64 == res["c"]
    assert np.dtype("<U") == res["d"]
    assert np.dtype("<M8[ns]") == res["e"]
    res = to_pandas_dtype(schema, True)
    assert pd.BooleanDtype() == res["a"]
    assert pd.Int32Dtype() == res["b"]
    assert pd.Float64Dtype() == res["c"]
    assert pd.StringDtype() == res["d"]
    assert np.dtype("<M8[ns]") == res["e"]

    schema2 = expression_to_schema("a:[bool],b:{c:long}")
    res = to_pandas_dtype(schema2, False)
    assert np.dtype("O") == res["a"]
    assert np.dtype("O") == res["b"]

    if hasattr(pd, "ArrowDtype"):
        res = to_pandas_dtype(schema, True, use_arrow_dtype=True)
        assert pd.BooleanDtype() == res["a"]
        assert pd.Int32Dtype() == res["b"]
        assert pd.Float64Dtype() == res["c"]
        assert pd.StringDtype() == res["d"]
        assert pd.ArrowDtype(schema[4].type) == res["e"]

        res = to_pandas_dtype(schema, False, use_arrow_dtype=True)
        assert pd.ArrowDtype(schema[0].type) == res["a"]
        assert pd.ArrowDtype(schema[1].type) == res["b"]
        assert pd.ArrowDtype(schema[2].type) == res["c"]
        assert pd.ArrowDtype(schema[3].type) == res["d"]
        assert pd.ArrowDtype(schema[4].type) == res["e"]


def test_pa_table_to_pandas():
    adf = pa.Table.from_pydict(
        {
            "a": [0, 1],
            "b": [[1, 2], [3, 4]],
            "c": ["a", "b"],
            "d": [{"x": "x"}, {"x": "x"}],
        },
    )
    pdf = pa_table_to_pandas(adf)
    assert pdf["a"].dtype == np.int32 or pdf["a"].dtype == np.int64
    assert pdf["b"].dtype == np.dtype("O")
    assert pdf["d"].dtype == np.dtype("O")
    pdf = pa_table_to_pandas(adf, use_extension_types=True)
    assert pdf["a"].dtype == pd.Int32Dtype() or pdf["a"].dtype == pd.Int64Dtype()
    assert pdf["b"].dtype == np.dtype("O")
    assert pdf["c"].dtype == pd.StringDtype()
    assert pdf["d"].dtype == np.dtype("O")

    if hasattr(pd, "ArrowDtype"):
        pdf = pa_table_to_pandas(adf, use_extension_types=False, use_arrow_dtype=True)
        assert pdf["a"].dtype == pd.ArrowDtype(pa.int64())
        assert pdf["b"].dtype == pd.ArrowDtype(pa.list_(pa.int64()))
        assert pdf["c"].dtype == pd.ArrowDtype(pa.string())
        assert pdf["d"].dtype == pd.ArrowDtype(pa.struct([pa.field("x", pa.string())]))
        pdf = pa_table_to_pandas(adf, use_extension_types=True, use_arrow_dtype=True)
        assert pdf["a"].dtype == pd.Int64Dtype()
        assert pdf["b"].dtype == pd.ArrowDtype(pa.list_(pa.int64()))
        assert pdf["c"].dtype == pd.StringDtype()
        assert pdf["d"].dtype == pd.ArrowDtype(pa.struct([pa.field("x", pa.string())]))


def test_is_supported():
    assert is_supported(pa.int32())
    assert is_supported(pa.decimal128(5, 2))
    assert is_supported(pa.timestamp("s"))
    assert is_supported(pa.date32())
    assert not is_supported(pa.date64())
    assert is_supported(pa.binary())
    assert not is_supported(pa.binary(0))
    assert not is_supported(pa.binary(1))
    assert is_supported(pa.struct([pa.field("a", pa.int32())]))
    assert is_supported(pa.list_(pa.int32()))
    assert is_supported(pa.map_(pa.int32(), pa.string()))
    raises(NotImplementedError, lambda: is_supported(pa.date64(), throw=True))


def test_get_eq_func():
    for t in [
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
    ]:
        assert not get_eq_func(t)(0, 1)
        assert not get_eq_func(t)(None, 1)
        assert get_eq_func(t)(1, 1)
        assert get_eq_func(t)(None, None)
    t = pa.null()
    assert get_eq_func(t)("0", "1")
    assert get_eq_func(t)(None, "1")
    assert get_eq_func(t)("1", "1")
    assert get_eq_func(t)(None, None)
    t = pa.string()
    assert not get_eq_func(t)("0", "1")
    assert not get_eq_func(t)(None, "1")
    assert get_eq_func(t)("1", "1")
    assert get_eq_func(t)(None, None)
    t = pa.bool_()
    assert not get_eq_func(t)(False, True)
    assert not get_eq_func(t)(None, False)
    assert not get_eq_func(t)(None, True)
    assert get_eq_func(t)(True, True)
    assert get_eq_func(t)(False, False)
    assert get_eq_func(t)(None, None)
    for t in [pa.float16(), pa.float32(), pa.float64()]:
        assert not get_eq_func(t)(0.0, 1.1)
        assert get_eq_func(t)(1.1, 1.1)
        assert get_eq_func(t)(None, float("nan"))
        for n in [None, float("nan"), float("inf"), float("-inf")]:
            assert not get_eq_func(t)(None, 1.1)
            assert get_eq_func(t)(None, None)
    for t in [pa.timestamp("ns"), pa.timestamp("us")]:
        for n in [None, pd.NaT]:
            assert not get_eq_func(t)(datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 1))
            assert not get_eq_func(t)(n, datetime(2020, 1, 1, 1))
            assert get_eq_func(t)(datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 1))
            assert get_eq_func(t)(n, n)
    assert get_eq_func(pa.timestamp("ns"))(None, pd.NaT)
    for t in [pa.date32()]:
        for n in [None, pd.NaT]:
            assert get_eq_func(t)(datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 1))
            assert not get_eq_func(t)(datetime(2020, 1, 1), datetime(2020, 1, 2).date())
            assert not get_eq_func(t)(n, datetime(2020, 1, 1, 1))
            assert get_eq_func(t)(datetime(2020, 1, 1).date(), datetime(2020, 1, 1, 1))
            assert get_eq_func(t)(n, n)
    t = pa.struct([pa.field("a", pa.int32())])
    assert not get_eq_func(t)(dict(a=0), dict(a=1))
    assert not get_eq_func(t)(None, dict(a=1))
    assert get_eq_func(t)(dict(a=1), dict(a=1))
    assert get_eq_func(t)(None, None)
    t = pa.list_(pa.int32())
    assert not get_eq_func(t)([0], [1])
    assert not get_eq_func(t)(None, [1])
    assert get_eq_func(t)([1], [1])
    assert get_eq_func(t)(None, None)
    t = pa.map_(pa.string(), pa.int32())
    assert not get_eq_func(t)({"a": 0}, {"a": 1})
    assert not get_eq_func(t)({"a": 0}, {"b": 0})
    assert not get_eq_func(t)(None, {"b": 0})
    assert get_eq_func(t)({"a": 0}, {"a": 0})
    assert get_eq_func(t)(None, None)


def test_schemaed_data_partitioner():
    p0 = SchemaedDataPartitioner(
        schema=expression_to_schema("a:int,b:int,c:int"),
        key_positions=[2, 0],
        row_limit=0,
    )
    p1 = SchemaedDataPartitioner(
        schema=expression_to_schema("a:int,b:int,c:int"),
        key_positions=[2, 0],
        row_limit=1,
    )
    p2 = SchemaedDataPartitioner(
        schema=expression_to_schema("a:int,b:int,c:int"),
        key_positions=[2, 0],
        row_limit=2,
    )
    data = [[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 0, 0]]
    _test_partition(p0, data, "0,0,[0,1,2];1,0,[3]")
    _test_partition(p1, data, "0,0,[0];0,1,[1];0,2,[2];1,0,[3]")
    _test_partition(p2, data, "0,0,[0,1];0,1,[2];1,0,[3]")
    _test_partition(p2, data, "0,0,[0,1];0,1,[2];1,0,[3]")  # can reuse the partitioner


def test_schemas_equal():
    a = expression_to_schema("a:int,b:int,c:int")
    b = expression_to_schema("a:int,b:int,c:int")
    c = expression_to_schema("a:int,c:int,b:int")
    d = expression_to_schema("a:int,b:int,c:long")
    assert schemas_equal(a, a)
    assert schemas_equal(a, b)
    assert not schemas_equal(a, c)
    assert schemas_equal(a, c, check_order=False)
    a = a.with_metadata({"a": "1"})
    assert schemas_equal(a, a)
    assert not schemas_equal(a, b)
    assert schemas_equal(a, b, check_metadata=False)
    assert not schemas_equal(a, c)
    assert not schemas_equal(a, c, check_order=False)
    assert not schemas_equal(a, c, check_metadata=False)
    assert schemas_equal(a, c, check_order=False, check_metadata=False)
    c = c.with_metadata({"a": "1"})
    assert not schemas_equal(a, c)
    assert schemas_equal(a, c, check_order=False)

    assert schemas_equal(a, d, ignore=[(pa.int32(), pa.int64())])
    assert schemas_equal(
        a, d, ignore=[(pa.int32(), pa.int16()), (pa.int64(), pa.int16())]
    )


def test_get_later_func():
    adf = pa.Table.from_pydict(
        {"a": [0, 1], "b": [2, 3]}, schema=expression_to_schema("a:int32,b:int32")
    )
    to_schema1 = expression_to_schema("a:int32,b:int32")
    to_schema2 = expression_to_schema("b:int32,a:long")
    to_schema3 = expression_to_schema("b:str,a:int32")
    to_schema4 = expression_to_schema("b:str")
    to_schema5 = expression_to_schema("b:str,d:str")

    f = get_alter_func(adf.schema, to_schema1, safe=True)
    tdf = f(adf)
    assert tdf.schema == to_schema1
    assert tdf.to_pydict() == {"a": [0, 1], "b": [2, 3]}

    f = get_alter_func(adf.schema, to_schema2, safe=True)
    tdf = f(adf)
    assert tdf.schema == to_schema2
    assert tdf.to_pydict() == {"b": [2, 3], "a": [0, 1]}

    f = get_alter_func(adf.schema, to_schema3, safe=True)
    tdf = f(adf)
    assert tdf.schema == to_schema3
    assert tdf.to_pydict() == {"b": ["2", "3"], "a": [0, 1]}

    f = get_alter_func(adf.schema, to_schema4, safe=True)
    tdf = f(adf)
    assert tdf.schema == to_schema4
    assert tdf.to_pydict() == {"b": ["2", "3"]}

    with raises(KeyError):
        get_alter_func(adf.schema, to_schema5, safe=True)

    adf = pa.Table.from_pydict(
        {"a": [datetime(2022, 1, 1), datetime(2022, 1, 2)], "b": ["a", "b"]},
        schema=pa.schema(
            [
                pa.field("a", pa.timestamp(unit="ns", tz="UTC")),
                pa.field("b", pa.large_string()),
            ]
        ),
    )
    to_schema10 = expression_to_schema("a:datetime,b:str")

    f = get_alter_func(adf.schema, to_schema10, safe=True)
    tdf = f(adf)
    assert tdf.schema == to_schema10
    assert tdf.to_pydict() == {
        "a": [datetime(2022, 1, 1), datetime(2022, 1, 2)],
        "b": ["a", "b"],
    }


def test_replace_type():
    ct = pa.string()
    assert ct is replace_type(
        ct, lambda t: pa.types.is_large_string(t), lambda t: pa.string()
    )  # no op
    assert pa.large_string() == replace_type(
        ct, lambda t: pa.types.is_string(t), lambda t: pa.large_string()
    )  # no op

    ct = pa.list_(pa.field("l", pa.int32()))
    assert ct is replace_type(ct, lambda t: pa.types.is_int64(t), lambda t: pa.int32())
    assert pa.list_(pa.field("l", pa.int64())) == replace_type(
        ct, lambda t: pa.types.is_int32(t), lambda t: pa.int64()
    )

    ct = pa.large_list(pa.field("l", pa.int32()))
    assert ct is replace_type(ct, lambda t: pa.types.is_int64(t), lambda t: pa.int32())
    assert pa.large_list(pa.field("l", pa.int64())) == replace_type(
        ct, lambda t: pa.types.is_int32(t), lambda t: pa.int64()
    )

    ct = pa.struct([pa.field("a", pa.struct([pa.field("b", pa.list_(pa.int32()))]))])
    assert ct is replace_type(
        ct, lambda t: pa.types.is_int64(t), lambda t: pa.int32()
    )  # no op
    assert pa.struct(
        [pa.field("a", pa.struct([pa.field("b", pa.list_(pa.int64()))]))]
    ) == replace_type(ct, lambda t: pa.types.is_int32(t), lambda t: pa.int64())
    assert ct is replace_type(
        ct, lambda t: pa.types.is_int32(t), lambda t: pa.int64(), recursive=False
    )

    ct = pa.map_(
        pa.field("l", pa.int32(), nullable=False), pa.field("m", pa.list_(pa.int32()))
    )
    assert ct is replace_type(ct, lambda t: pa.types.is_int64(t), lambda t: pa.int32())
    assert pa.map_(
        pa.field("l", pa.int64(), nullable=False), pa.field("m", pa.list_(pa.int64()))
    ) == replace_type(ct, lambda t: pa.types.is_int32(t), lambda t: pa.int64())
    assert ct is replace_type(
        ct, lambda t: pa.types.is_int32(t), lambda t: pa.int64(), recursive=False
    )

    # list conversion
    ct = pa.large_list(
        pa.field("l", pa.struct([pa.field("m", pa.large_list(pa.int32()))]))
    )
    assert ct is replace_type(
        ct, pa.types.is_list, lambda t: pa.large_list(t.value_field)
    )
    assert pa.list_(
        pa.field("l", pa.struct([pa.field("m", pa.list_(pa.int32()))]))
    ) == replace_type(ct, pa.types.is_large_list, lambda t: pa.list_(t.value_field))


def test_replace_types_in_schema():
    def _test(schema, from_type, to_type, expected, recursive=True):
        assert expression_to_schema(expected) == replace_types_in_schema(
            expression_to_schema(schema),
            [(to_pa_datatype(from_type), to_pa_datatype(to_type))],
            recursive=recursive,
        )

    def _same(schema, from_type, to_type):
        orig = expression_to_schema(schema)
        assert orig is replace_types_in_schema(
            orig, [(to_pa_datatype(from_type), to_pa_datatype(to_type))]
        )

    _same("a:int,b:int,c:[int]", "int", "int")
    _same("a:int,b:int,c:[int]", "int", "int")
    _same("a:int,b:int,c:[int]", "long", "int")
    _same("a:int,b:int,c:[int]", "long", "long")
    _same("a:int,b:int,c:[int]", "str", "str")

    _test("a:int,b:str,c:int", "int", "long", "a:long,b:str,c:long")
    _test("a:int,b:int,c:[int]", "int", "long", "a:long,b:long,c:[long]")
    _test(
        "a:int,b:int,c:[int]", "int", "long", "a:long,b:long,c:[int]", recursive=False
    )
    _test("a:{a:[int],b:<int,long>}", "int", "long", "a:{a:[long],b:<long,long>}")


@pytest.mark.skipif(PYARROW_VERSION.major < 11, reason="requires pyarrow>=11")
def test_replace_types_in_table():
    df = pa.Table.from_arrays(
        [[1], ["sadf"], [["a", "b"]]],
        schema=pa.schema(
            [
                ("a", pa.int32()),
                ("b", pa.large_string()),
                ("c", pa.list_(pa.field("l", pa.large_string()))),
            ]
        ),
    )
    assert df is replace_types_in_table(df, [(pa.int32(), pa.int32())])
    assert df is replace_types_in_table(df, [(pa.int64(), pa.int32())])
    assert df is replace_types_in_table(df, [(pa.string(), pa.large_string())])
    assert df is replace_types_in_table(
        df, [(pa.types.is_large_list, lambda t: pa.list_(t.value_field))]
    )
    assert replace_types_in_table(df, [(pa.int32(), pa.int64())]).schema == pa.schema(
        [
            ("a", pa.int64()),
            ("b", pa.large_string()),
            ("c", pa.list_(pa.field("x", pa.large_string()))),
        ]
    )
    assert replace_types_in_table(
        df, [(pa.large_string(), pa.string())]
    ).schema == pa.schema(
        [
            ("a", pa.int32()),
            ("b", pa.string()),
            ("c", pa.list_(pa.field("x", pa.string()))),
        ]
    )
    assert replace_types_in_table(
        df, [(pa.large_string(), pa.string())], recursive=False
    ).schema == pa.schema(
        [
            ("a", pa.int32()),
            ("b", pa.string()),
            ("c", pa.list_(pa.field("x", pa.large_string()))),
        ]
    )
    assert replace_types_in_table(
        df,
        [
            (pa.large_string(), pa.string()),
            (pa.types.is_list, lambda t: pa.large_list(t.value_field)),
        ],
    ).schema == pa.schema(
        [
            ("a", pa.int32()),
            ("b", pa.string()),
            ("c", pa.large_list(pa.field("x", pa.string()))),
        ]
    )


@pytest.mark.skipif(PYARROW_VERSION.major < 11, reason="requires pyarrow>=11")
def test_replace_large_types():
    warnings.filterwarnings("ignore")
    df = pa.Table.from_arrays(
        [[b"x"], ["sadf"], [{"a": ["b"]}]],
        schema=pa.schema(
            [
                ("a", pa.large_binary()),
                ("b", pa.large_string()),
                (
                    "c",
                    pa.struct(
                        [pa.field("l", pa.large_list(pa.field("m", pa.large_string())))]
                    ),
                ),
            ]
        ),
    )
    assert replace_types_in_table(df, LARGE_TYPES_REPLACEMENT).schema == pa.schema(
        [
            ("a", pa.binary()),
            ("b", pa.string()),
            (
                "c",
                pa.struct([pa.field("l", pa.list_(pa.field("m", pa.string())))]),
            ),
        ]
    )


def test_cast_pa_table():
    df = pa.Table.from_arrays(
        [pa.array([1, 2]), pa.array(["true", None])],
        schema=pa.schema([("a", pa.int32()), ("b", pa.string())]),
    )
    assert df is cast_pa_table(df, df.schema)
    assert cast_pa_table(df, expression_to_schema("a:str,b:bool")).to_pydict() == {
        "a": ["1", "2"],
        "b": [True, None],
    }


def test_cast_pa_array():
    def _test(arr, orig_type, new_type, expected=None, exact=True):
        x = pa.array(arr, type=orig_type)
        res = cast_pa_array(x, new_type)
        assert res.type == new_type
        if exact:
            assert res.to_pylist() == (expected or arr)
        else:
            for a, e in zip(res.to_pylist(), (expected or arr)):
                if pd.isna(e):
                    assert pd.isna(a)
                else:
                    assert abs(a - e) < 1e-6

    # bytes -> bytes
    _test([b"abc", None], pa.binary(), pa.binary())

    # int -> int
    _test([1, 2, None], pa.int32(), pa.int32())
    _test([1, 2, None], pa.int32(), pa.int64())
    with raises(Exception):
        _test([1, 200000, None], pa.int64(), pa.int8())

    # int -> float
    _test([1, 2, None], pa.int32(), pa.float32(), [1.0, 2.0, None])

    # int -> bool
    _test([0, 1, None], pa.int32(), pa.bool_(), [False, True, None])
    _test([0, 2, None], pa.int32(), pa.bool_(), [False, True, None])

    # int -> str
    _test([0, 1, None], pa.int32(), pa.string(), ["0", "1", None])

    # float -> int
    _test([1.0, 2.0, None], pa.float32(), pa.int32(), [1, 2, None])
    _test([1.1, 2.2, None], pa.float32(), pa.int64(), [1, 2, None])

    # float -> float
    _test([1.1, 2.2, None], pa.float32(), pa.float32(), [1.1, 2.2, None], exact=False)
    _test(
        [1.1, 2.2, float("nan")],
        pa.float32(),
        pa.float32(),
        [1.1, 2.2, None],
        exact=False,
    )

    # float -> bool
    _test([0.0, 1.1, None], pa.float32(), pa.bool_(), [False, True, None])

    # float -> str
    _test([0.1, 1.1, None], pa.float32(), pa.string(), ["0.1", "1.1", None])

    # bool -> int
    _test([True, False, None], pa.bool_(), pa.int32(), [1, 0, None])

    # bool -> float
    _test([True, False, None], pa.bool_(), pa.float32(), [1.0, 0.0, None], exact=False)

    # bool -> bool
    _test([True, False, None], pa.bool_(), pa.bool_(), [True, False, None])

    # bool -> str
    _test([True, False, None], pa.bool_(), pa.string(), ["true", "false", None])

    # str -> int
    _test(["1", "2", None], pa.string(), pa.int32(), [1, 2, None])

    # str -> float
    _test(["1", "2", None], pa.string(), pa.float32(), [1.0, 2.0, None], exact=False)
    _test(
        ["1.1", "2.1", None], pa.string(), pa.float32(), [1.1, 2.1, None], exact=False
    )

    # str -> bool
    _test(["trUe", "False", None], pa.string(), pa.bool_(), [True, False, None])

    # str -> str
    _test(["a", "b", None], pa.string(), pa.string())

    # str -> datetime no tz
    _test(
        ["2023-01-03 00:01:02", None],
        pa.string(),
        pa.timestamp("ns"),
        [datetime(2023, 1, 3, 0, 1, 2), None],
    )

    # str -> datetime + tz
    _test(
        ["2023-01-03 00:01:02-0800", None],
        pa.string(),
        pa.timestamp("ns", tz="-08:00"),
        [pd.Timestamp("2023-01-03 00:01:02-0800", tz="-08:00"), None],
    )

    # str -> date
    _test(
        ["2023-01-03", None],
        pa.string(),
        pa.date32(),
        [date(2023, 1, 3), None],
    )

    # datetime -> str
    _test(
        [datetime(2023, 1, 2, 3, 4), None],
        pa.timestamp("ns"),
        pa.string(),
        ["2023-01-02 03:04:00", None],
    )

    _test(
        [datetime(2023, 1, 2, 3, 4), datetime(2023, 1, 2, 3, 4, 5, 6), None],
        pa.timestamp("ns"),
        pa.string(),
        ["2023-01-02 03:04:00.000000", "2023-01-02 03:04:05.000006", None],
    )

    # datetime -> datetime
    _test(
        [datetime(2023, 1, 2, 3, 4), None],
        pa.timestamp("ns"),
        pa.timestamp("ns"),
        [datetime(2023, 1, 2, 3, 4), None],
    )

    # datetime -> datetime+tz
    _test(
        [datetime(2023, 1, 2, 3, 4), None],
        pa.timestamp("ns"),
        pa.timestamp("ns", tz="US/Pacific"),
        [pd.Timestamp("2023-01-02 03:04:00", tz="US/Pacific"), None],
    )

    # datetime -> date
    _test(
        [datetime(2023, 1, 2, 3, 4), None],
        pa.timestamp("ns"),
        pa.date32(),
        [date(2023, 1, 2), None],
    )

    # datetime+tz -> str
    _test(
        [pd.Timestamp("2023-01-03 00:03:04", tz="US/Pacific"), None],
        pa.timestamp("ns", tz="US/Pacific"),
        pa.string(),
        ["2023-01-03 00:03:04-08:00", None],
    )

    # datetime+tz -> datetime
    _test(
        [pd.Timestamp("2023-01-03 00:03:04", tz="US/Pacific"), None],
        pa.timestamp("ns", tz="US/Pacific"),
        pa.timestamp("ns"),
        [pd.Timestamp("2023-01-03 00:03:04"), None],
    )

    # datetime+tz -> datetime+tz
    _test(
        [pd.Timestamp("2023-01-03 00:03:04", tz="US/Pacific"), None],
        pa.timestamp("ns", tz="US/Pacific"),
        pa.timestamp("ns", tz="UTC"),
        [pd.Timestamp("2023-01-03 08:03:04", tz="UTC"), None],
    )

    # datetime+tz -> date
    _test(
        [pd.Timestamp("2023-01-03 00:03:04", tz="US/Pacific"), None],
        pa.timestamp("ns", tz="US/Pacific"),
        pa.date32(),
        [date(2023, 1, 3), None],
    )

    # datetime+tz -> str
    _test(
        [pd.Timestamp("2023-01-03 00:03:04", tz="US/Pacific"), None],
        pa.timestamp("ns", tz="US/Pacific"),
        pa.string(),
        ["2023-01-03 00:03:04-08:00", None],
    )

    # date -> str
    _test(
        [date(2023, 1, 2), None],
        pa.date32(),
        pa.string(),
        ["2023-01-02", None],
    )

    # date -> datetime
    _test(
        [date(2023, 1, 2), None],
        pa.date32(),
        pa.timestamp("ns"),
        [datetime(2023, 1, 2), None],
    )

    # date -> datetime+tz
    _test(
        [date(2023, 1, 2), None],
        pa.date32(),
        pa.timestamp("ns", tz="US/Pacific"),
        [pd.Timestamp("2023-01-02 00:00:00-0800", tz="US/Pacific"), None],
    )

    # date -> date
    _test(
        [date(2023, 1, 2), None],
        pa.date32(),
        pa.date64(),
        [date(2023, 1, 2), None],
    )


def _test_partition(partitioner, data, expression):
    e = []
    for p, s, ck in partitioner.partition(data):
        idx = [data.index(x) for x in ck]
        e.append(f"{p},{s},{idx}")
    assert expression == ";".join(e).replace(" ", "")


def _assert_from_expr(expr, expected=None):
    schema = expression_to_schema(expr)
    out_expr = schema_to_expression(schema)
    expected = expected or expr
    assert expected == out_expr
