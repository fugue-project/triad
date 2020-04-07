from datetime import date, datetime

import numpy as np
import pandas as pd
import pyarrow as pa
from pytest import raises
from triad.utils.pyarrow import (_parse_type, _type_to_expression,
                                 expression_to_schema, is_supported,
                                 pandas_to_schema, schema_to_expression,
                                 to_pa_datatype, validate_column_name)


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
    _assert_from_expr("a : { x : int32 , y : [string] } , b : [ uint8 ] ",
                      "a:{x:int,y:[str]},b:[ubyte]")
    _assert_from_expr("a : [{ x : int32 , y : [string] }] , b : [ uint8 ] ",
                      "a:[{x:int,y:[str]}],b:[ubyte]")
    _assert_from_expr("a:decimal(5,2)")

    raises(SyntaxError, lambda: expression_to_schema("123:int"))
    raises(SyntaxError, lambda: expression_to_schema("int"))
    raises(SyntaxError, lambda: expression_to_schema("a:dummytype"))
    raises(SyntaxError, lambda: expression_to_schema("a:int,a:str"))
    raises(SyntaxError, lambda: expression_to_schema("a:int,b:{x:int,x:str}"))
    raises(SyntaxError, lambda: expression_to_schema("_:int"))
    raises(SyntaxError, lambda: expression_to_schema("__:int"))


def test__parse_type():
    assert pa.int32() == _parse_type(" int ")
    assert pa.timestamp("ns") == _parse_type(" datetime ")
    assert pa.timestamp(
        "s", 'America/New_York') == _parse_type(" timestamp ( s , America/New_York ) ")
    assert pa.timestamp("s") == _parse_type(" timestamp ( s ) ")
    assert _parse_type(" timestamp ( ns ) ") == _parse_type(" datetime ")
    assert pa.decimal128(5, 2) == _parse_type(" decimal(5,2) ")
    assert pa.decimal128(5) == _parse_type(" decimal ( 5 )  ")


def test__type_to_expression():
    assert "int" == _type_to_expression(pa.int32())
    assert "datetime" == _type_to_expression(pa.timestamp("ns"))
    assert "timestamp(ns,America/New_York)" == _type_to_expression(
        pa.timestamp("ns", "America/New_York"))
    assert "timestamp(s)" == _type_to_expression(
        pa.timestamp("s"))
    assert "decimal(5)" == _type_to_expression(pa.decimal128(5))
    assert "decimal(5,2)" == _type_to_expression(pa.decimal128(5, 2))
    raises(NotImplementedError, lambda: _type_to_expression(pa.large_binary()))


def test_to_pa_datatype():
    assert pa.int32() == to_pa_datatype(pa.int32())
    assert pa.int32() == to_pa_datatype("int")
    assert pa.int64() == to_pa_datatype(int)
    assert pa.float64() == to_pa_datatype(float)
    assert pa.float64() == to_pa_datatype(np.float64)
    assert pa.timestamp("ns") == to_pa_datatype(datetime)
    assert pa.date32() == to_pa_datatype(date)
    assert pa.date32() == to_pa_datatype("date")
    raises(TypeError, lambda: to_pa_datatype(123))
    raises(TypeError, lambda: to_pa_datatype(None))


def test_is_supported():
    assert is_supported(pa.int32())
    assert is_supported(pa.decimal128(5, 2))
    assert is_supported(pa.timestamp("s"))
    assert is_supported(pa.date32())
    assert not is_supported(pa.date64())
    assert not is_supported(pa.binary())
    assert is_supported(pa.struct([pa.field("a", pa.int32())]))
    assert is_supported(pa.list_(pa.int32()))


def test_pandas_to_schema():
    df = pd.DataFrame([[1.0, 2], [2.0, 3]])
    raises(ValueError, lambda: pandas_to_schema(df))
    df = pd.DataFrame([[1.0, 2], [2.0, 3]], columns=["x", "y"])
    assert list(pa.Schema.from_pandas(df)) == list(pandas_to_schema(df))
    df = pd.DataFrame([["a", 2], ["b", 3]], columns=["x", "y"])
    assert list(pa.Schema.from_pandas(df)) == list(pandas_to_schema(df))
    df = pd.DataFrame([], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype('object')})
    assert [pa.field("x", pa.int32()), pa.field(
        "y", pa.string())] == list(pandas_to_schema(df))
    df = pd.DataFrame([[1, "x"], [2, "y"]], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype('object')})
    assert list(pa.Schema.from_pandas(df)) == list(pandas_to_schema(df))
    df = pd.DataFrame([[1, "x"], [2, "y"]], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype(str)})
    assert list(pa.Schema.from_pandas(df)) == list(pandas_to_schema(df))
    df = pd.DataFrame([[1, "x"], [2, "y"]], columns=["x", "y"])
    df = df.astype(dtype={"x": np.int32, "y": np.dtype('str')})
    assert list(pa.Schema.from_pandas(df)) == list(pandas_to_schema(df))


def _assert_from_expr(expr, expected=None):
    schema = expression_to_schema(expr)
    out_expr = schema_to_expression(schema)
    expected = expected or expr
    assert expected == out_expr
