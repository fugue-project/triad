from datetime import datetime
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import as_type
from triad.utils.json import loads_no_dup

_TYPE_EXPRESSION_MAPPING: Dict[str, pa.DataType] = {
    "str": pa.string(),
    "string": pa.string(),
    "bool": pa.bool_(),
    "boolean": pa.bool_(),
    "int8": pa.int8(),
    "byte": pa.int8(),
    "int16": pa.int16(),
    "short": pa.int16(),
    "int32": pa.int32(),
    "int": pa.int32(),
    "long": pa.int64(),
    "int64": pa.int64(),
    "uint8": pa.uint8(),
    "ubyte": pa.uint8(),
    "uint16": pa.uint16(),
    "ushort": pa.uint16(),
    "uint32": pa.uint32(),
    "uint": pa.uint32(),
    "ulong": pa.uint64(),
    "uint64": pa.uint64(),
    "float16": pa.float16(),
    "float": pa.float32(),
    "float32": pa.float32(),
    "double": pa.float64(),
    "float64": pa.float64(),
    "datetime": pa.timestamp("ns"),
}

_TYPE_EXPRESSION_R_MAPPING: Dict[pa.DataType, str] = {
    pa.string(): "str",
    pa.bool_(): "bool",
    pa.int8(): "byte",
    pa.int16(): "short",
    pa.int32(): "int",
    pa.int64(): "long",
    pa.uint8(): "ubyte",
    pa.uint16(): "ushort",
    pa.uint32(): "uint",
    pa.uint64(): "ulong",
    pa.float16(): "float16",
    pa.float32(): "float",
    pa.float64(): "double",
    pa.timestamp("ns"): "datetime",
}


_SPECIAL_TOKENS: Set[str] = {",", "{", "}", "[", "]", ":"}


def validate_column_name(expr: str) -> bool:
    """Check if `expr` is a valid column name based on Triad standard: it has to be
    a valid python identifier plus it can't be purly `_`

    :param expr: column name expression
    :return: whether it is valid
    """
    if expr is None or not expr.isidentifier():
        return False
    return expr.strip("_") != ""


def expression_to_schema(expr: str) -> pa.Schema:
    """Convert schema expression to `pyarrow.Schema`.

    Format: `col_name:col_type[,col_name:col_type]+`

    If col_type is a list type, the syntax should be `[element_type]`

    If col_type is a struct type, the syntax should
    be `{col_name:col_type[,col_name:col_type]+}`

    Whitespaces will be removed. The format of the expression is json
    without any double quotes

    :Examples:
    >>> expression_to_schema("a:int,b:int")
    >>> expression_to_schema("a:[int],b:{x:int,y:{z:[str],w:byte}}")

    :param expr: schema expression
    :raises SyntaxError: if there is syntax issue or unknown types
    :return: pyarrow.Schema
    """
    try:
        json_str = "{" + "".join(list(_parse_tokens(expr))[:-1]) + "}"
        obj = loads_no_dup(json_str)
        return pa.schema(_construct_struct(obj))
    except SyntaxError:
        raise
    except Exception as e:
        raise SyntaxError(f"Invalid syntax: '{expr}' {str(e)}")


def schema_to_expression(schema: pa.Schema) -> pa.Schema:
    """Convert pyarrow.Schema to Triad schema expression
    see :func:~triad.utils.pyarrow.expression_to_schema""

    :param schema: pyarrow schema
    :raises NotImplementedError: if there some type is not supported by Triad
    :return: schema string expression
    """
    return ",".join(_field_to_expression(x) for x in list(schema))


def to_pa_datatype(obj: Any) -> pa.DataType:
    """Convert an object to pyarrow DataType

    :param obj: any object
    :raises TypeError: if unable to convert
    :return: an instance of pd.DataType
    """
    if isinstance(obj, pa.DataType):
        return obj
    if isinstance(obj, str):
        return _parse_type(obj)
    if issubclass(obj, datetime):
        return pa.timestamp("ns")
    return pa.from_numpy_dtype(np.dtype(obj))


def is_supported(data_type: pa.DataType) -> bool:
    """Whether `data_type` is currently supported by Triad

    :param data_type: instance of pa.DataType
    :return: if it is supported
    """
    if data_type in _TYPE_EXPRESSION_R_MAPPING:
        return True
    return isinstance(
        data_type, (pa.Decimal128Type, pa.TimestampType, pa.StructType, pa.ListType)
    )


def _to_pytype(pytype: type, obj: Any) -> Any:
    if obj is None or isinstance(obj, pytype):
        return obj
    if obj != obj:  # NaN
        return None
    return as_type(obj, pytype)


def _to_pydecimal(obj: Any) -> Any:
    if obj is None or isinstance(obj, float):
        return obj
    if obj != obj:  # NaN
        return None
    return as_type(obj, float)


def _to_pydatetime(obj: Any) -> Any:
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, datetime):
        return obj
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    if obj != obj:  # NaN
        return None
    return as_type(obj, datetime)


def _assert_pytype(pytype: type, obj: Any) -> Any:
    assert isinstance(obj, pytype)
    return obj


_PATYPE_TO_PYTYPE_CONVERTERS: Dict[pa.DataType, Any] = {
    pa.string(): lambda x: _to_pytype(str, x),
    pa.bool_(): lambda x: _to_pytype(bool, x),
    pa.int8(): lambda x: _to_pytype(int, x),
    pa.int16(): lambda x: _to_pytype(int, x),
    pa.int32(): lambda x: _to_pytype(int, x),
    pa.int64(): lambda x: _to_pytype(int, x),
    pa.uint8(): lambda x: _to_pytype(int, x),
    pa.uint16(): lambda x: _to_pytype(int, x),
    pa.uint32(): lambda x: _to_pytype(int, x),
    pa.uint64(): lambda x: _to_pytype(int, x),
    pa.float16(): lambda x: _to_pytype(float, x),
    pa.float32(): lambda x: _to_pytype(float, x),
    pa.float64(): lambda x: _to_pytype(float, x),
}


class TypeConverter(object):
    def __init__(self, schema: pa.Schema):
        self._schema = schema
        self._to_pytype = [_no_op_convert] * len(schema)
        self._build_to_pytype()
        # self._to_dtype = [_no_op_convert] * len(schema)

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def _build_to_pytype(self) -> None:
        for i in range(len(self.schema)):
            f = self.schema.get_field_index(i)
            if f.type in _PATYPE_TO_PYTYPE_CONVERTERS:
                self._to_pytype[i] = _PATYPE_TO_PYTYPE_CONVERTERS[f.type]
            elif isinstance(f.type, pa.TimestampType):
                self._to_pytype[i] = lambda x: _to_pydatetime(x)
            elif isinstance(f.type, pa.Decimal128Type):
                self._to_pytype[i] = lambda x: _to_pydecimal(x)
            elif isinstance(f.type, pa.StructType):
                self._to_pytype[i] = lambda x: _assert_pytype(dict, x)
            elif isinstance(f.type, pa.ListType):
                self._to_pytype[i] = lambda x: _assert_pytype(list, x)


def _field_to_expression(field: pa.Field) -> str:
    return f"{field.name}:{_type_to_expression(field.type)}"


def _type_to_expression(dt: pa.DataType) -> str:
    if dt in _TYPE_EXPRESSION_R_MAPPING:
        return _TYPE_EXPRESSION_R_MAPPING[dt]
    if isinstance(dt, pa.TimestampType):
        if dt.tz is None:
            return f"timestamp({dt.unit})"
        else:
            return f"timestamp({dt.unit},{dt.tz})"
    if isinstance(dt, pa.Decimal128Type):
        if dt.scale == 0:
            return f"decimal({dt.precision})"
        else:
            return f"decimal({dt.precision},{dt.scale})"
    if isinstance(dt, pa.StructType):
        f = ",".join(_field_to_expression(x) for x in list(dt))
        return "{" + f + "}"
    if isinstance(dt, pa.ListType):
        f = _type_to_expression(dt.value_type)
        return "[" + f + "]"
    raise NotImplementedError(f"{dt} is not supported")


def _construct_struct(obj: Dict[str, Any]) -> Iterable[pa.Field]:
    for k, v in obj.items():
        if not validate_column_name(k):
            raise SyntaxError(f"{k} is not a valid field name")
        if isinstance(v, str):
            yield pa.field(k, _parse_type(v))
        elif isinstance(v, Dict):
            yield pa.field(k, pa.struct(_construct_struct(v)))
        elif isinstance(v, List):
            assert_or_throw(1 == len(v), SyntaxError(f"{v} is not a valid list type"))
            yield pa.field(k, pa.list_(_parse_type(v[0])))
        else:  # pragma: no cover
            raise SyntaxError(f"{k} {v} is not a valid field")


def _parse_type(expr: str) -> pa.DataType:
    name, args = _parse_type_function(expr)
    if name in _TYPE_EXPRESSION_MAPPING:
        assert len(args) == 0, f"{expr} can't have arguments"
        return _TYPE_EXPRESSION_MAPPING[name]
    # else: decimal(precision)
    if name == "decimal":
        assert 1 <= len(args) <= 2, f"{expr}, decimal must have 1 or 2 argument"
        return pa.decimal128(int(args[0]), 0 if len(args) == 1 else int(args[1]))
    if name == "timestamp":
        assert 1 <= len(args) <= 2, f"{expr}, timestamp must have 1 or 2 arguments"
        return pa.timestamp(args[0], None if len(args) == 1 else args[1])
    raise SyntaxError(f"{expr} is not a supported type")


def _parse_type_function(expr: str) -> Tuple[str, List[str]]:
    p = expr.split("(", 1)
    name = p[0].strip()
    assert validate_column_name(name), f"Invalid expression {expr}"
    if len(p) == 1:
        return name, []
    arg_expr = p[1].strip().rstrip(")")
    args = [x.strip() for x in arg_expr.split(",")]
    return name, args


def _parse_tokens(expr: str) -> Iterable[str]:
    expr += ","
    last = 0
    for i in range(len(expr)):
        if expr[i] in _SPECIAL_TOKENS:
            s = expr[last:i].strip()
            if s != "":
                yield '"' + s + '"'
            yield expr[i]
            last = i + 1


def _no_op_convert(obj: Any) -> Any:
    return obj
