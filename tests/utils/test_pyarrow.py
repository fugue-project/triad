from datetime import date, datetime

import numpy as np
import pandas as pd
import pyarrow as pa
from pytest import raises
from triad.utils.pyarrow import (
    TRIAD_DEFAULT_TIMESTAMP,
    SchemaedDataPartitioner,
    _parse_type,
    _type_to_expression,
    expression_to_schema,
    get_eq_func,
    is_supported,
    schema_to_expression,
    schemas_equal,
    to_pa_datatype,
    to_pandas_dtype,
    to_single_pandas_dtype,
    validate_column_name,
)


def test_validate_column_name():
    assert validate_column_name("abc")
    assert validate_column_name("__abc__")
    assert validate_column_name("_1_")
    assert not validate_column_name(None)
    assert not validate_column_name("")
    assert not validate_column_name("_")
    assert not validate_column_name("__")
    assert not validate_column_name("1")
    assert not validate_column_name("a ")


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

    raises(SyntaxError, lambda: expression_to_schema("123:int"))
    raises(SyntaxError, lambda: expression_to_schema("int"))
    raises(SyntaxError, lambda: expression_to_schema("a:dummytype"))
    raises(SyntaxError, lambda: expression_to_schema("a:int,a:str"))
    raises(SyntaxError, lambda: expression_to_schema("a:int,b:{x:int,x:str}"))
    raises(SyntaxError, lambda: expression_to_schema("_:int"))
    raises(SyntaxError, lambda: expression_to_schema("__:int"))
    raises(SyntaxError, lambda: expression_to_schema("a:[int,str]"))
    raises(SyntaxError, lambda: expression_to_schema("a:[]"))
    raises(SyntaxError, lambda: expression_to_schema("a:<>"))
    raises(SyntaxError, lambda: expression_to_schema("a:<int>"))
    raises(SyntaxError, lambda: expression_to_schema("a:<int,str,str>"))


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
    assert pd.UInt32Dtype() == to_single_pandas_dtype(pa.uint32(), True)
    assert np.float32 == to_single_pandas_dtype(pa.float32(), True)
    assert pd.StringDtype() == to_single_pandas_dtype(pa.string(), True)
    assert np.dtype("<M8[ns]") == to_single_pandas_dtype(pa.timestamp("ns"), True)


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
    assert np.float64 == res["c"]
    assert pd.StringDtype() == res["d"]
    assert np.dtype("<M8[ns]") == res["e"]


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
