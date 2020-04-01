import pyarrow as pa
from pytest import raises
from triad.utils.pyarrow import (_parse_type, _type_to_expression,
                                 expression_to_schema, schema_to_expression,
                                 validate_column_name)


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


def _assert_from_expr(expr, expected=None):
    schema = expression_to_schema(expr)
    out_expr = schema_to_expression(schema)
    expected = expected or expr
    assert expected == out_expr
