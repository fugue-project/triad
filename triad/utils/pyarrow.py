import json
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import as_type
from triad.utils.json import loads_no_dup
from triad.utils.string import validate_triad_var_name


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
    "date": pa.date32(),
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
    pa.date32(): "date",
    pa.timestamp("ns"): "datetime",
}


_SPECIAL_TOKENS: Set[str] = {",", "{", "}", "[", "]", ":"}


def validate_column_name(expr: str) -> bool:
    """Check if `expr` is a valid column name.
    See :func:`~triad.utils.string.validate_triad_var_name`

    :param expr: column name expression
    :return: whether it is valid
    """
    return validate_triad_var_name(expr)


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
    if issubclass(obj, date):
        return pa.date32()
    return pa.from_numpy_dtype(np.dtype(obj))


def pandas_to_schema(df: pd.DataFrame) -> pa.Schema:
    """Extract pandas dataframe schema as pyarrow schema. This is a replacement
    of pyarrow.Schema.from_pandas, and it can correctly handle string type and
    empty dataframes

    :param df: pandas dataframe
    :raises ValueError: if pandas dataframe does not have named schema
    :return: pyarrow.Schema
    """
    if df.columns.dtype != "object":
        raise ValueError("Pandas dataframe must have named schema")
    if df.shape[0] > 0:
        return pa.Schema.from_pandas(df)
    fields: List[pa.Field] = []
    for i in range(df.shape[1]):
        tp = df.dtypes[i]
        if tp == np.dtype("object") or tp == np.dtype(str):
            t = pa.string()
        else:
            t = pa.from_numpy_dtype(tp)
        fields.append(pa.field(df.columns[i], t))
    return pa.schema(fields)


def is_supported(data_type: pa.DataType) -> bool:
    """Whether `data_type` is currently supported by Triad

    :param data_type: instance of pa.DataType
    :return: if it is supported
    """
    if data_type in _TYPE_EXPRESSION_R_MAPPING:
        return True
    tp = (pa.Decimal128Type, pa.TimestampType, pa.StructType, pa.ListType)
    return isinstance(data_type, tp)


def apply_schema(
    schema: pa.Schema,
    data: Iterable[List[Any]],
    copy: bool = True,
    deep: bool = False,
    str_as_json: bool = True,
) -> Iterable[List[Any]]:
    """Use `pa.Schema` to convert a row(list) to the correspondent types.

    Notice this function is to convert from python native type to python
    native type. It is used to normalize data input, which could be generated
    by different logics, into the correct data types.

    Notice this function assumes each item of `data` has the same length
    with `schema` and will not do any extra validation on that.

    :param schema: pyarrow schema
    :param data: and iterable of rows, represtented by list or tuple
    :param copy: whether to apply inplace (copy=False), or create new instances
    :param deep: whether to do deep conversion on nested (struct, list) types
    :param str_as_json: where to treat string data as json for nested types

    :raises ValueError: if any value can't be converted to the datatype
    :raises NotImplementedError: if any field type is not supported by Triad

    :yield: converted rows
    """
    converter = _TypeConverter(schema, copy=copy, deep=deep, str_as_json=str_as_json)
    try:
        for item in data:
            yield converter.row_to_py(item)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(str(e))


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
            if isinstance(v[0], str):
                yield pa.field(k, pa.list_(_parse_type(v[0])))
            else:
                yield pa.field(k, pa.list_(pa.struct(_construct_struct(v[0]))))
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
    skip = False
    for i in range(len(expr)):
        if expr[i] == "(":
            skip = True
        if expr[i] == ")":
            skip = False
        if not skip and expr[i] in _SPECIAL_TOKENS:
            s = expr[last:i].strip()
            if s != "":
                yield '"' + s + '"'
            yield expr[i]
            last = i + 1


def _to_pyint(obj: Any) -> Any:
    if obj is None or isinstance(obj, int):
        return obj
    if obj != obj:  # NaN
        return None
    return as_type(obj, int)


def _to_pystr(obj: Any) -> Any:
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _to_pybool(obj: Any) -> Any:
    if obj is None or isinstance(obj, bool):
        return obj
    return as_type(obj, bool)


def _to_pyfloat(obj: Any) -> Any:
    if obj is None or isinstance(obj, float):
        return obj
    if obj != obj:  # NaN
        return None
    return as_type(obj, float)


def _to_pydatetime(obj: Any) -> Any:
    if obj is None or obj is pd.NaT:
        return pd.NaT
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    if isinstance(obj, datetime):
        return obj
    return as_type(obj, datetime)


def _to_pydate(obj: Any) -> Any:
    if obj is None or obj is pd.NaT:
        return pd.NaT
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime().date()
    if isinstance(obj, datetime):
        return obj.date()
    if isinstance(obj, date):
        return obj
    return as_type(obj, datetime).date()


def _assert_pytype(pytype: type, obj: Any) -> Any:
    if obj is None or isinstance(obj, pytype):
        return obj
    raise TypeError(f"{obj} is not {pytype}")


def _to_pydict(
    schema: Dict[str, Callable[[Any], Any]], obj: Any, str_as_json: bool = True
) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str) and str_as_json:
        obj = json.loads(obj)
    if not isinstance(obj, Dict):
        raise TypeError(f"{obj} is not dict")
    result: Dict[str, Any] = {}
    for k, v in schema.items():
        result[k] = v(obj.get(k, None))
    return result


def _to_pylist(
    converter: Callable[[Any], Any],
    obj: Any,
    copy: bool = True,
    str_as_json: bool = True,
) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str) and str_as_json:
        obj = json.loads(obj)
    if not isinstance(obj, List):
        raise TypeError(f"{obj} is not list")
    if copy:
        return [converter(x) for x in obj]
    else:
        for i in range(len(obj)):
            obj[i] = converter(obj[i])
        return obj


def _no_op_convert(obj: Any) -> Any:  # pragma: no cover
    return obj


class _TypeConverter(object):
    _CONVERTERS: Dict[pa.DataType, Any] = {
        pa.string(): _to_pystr,
        pa.bool_(): _to_pybool,
        pa.int8(): _to_pyint,
        pa.int16(): _to_pyint,
        pa.int32(): _to_pyint,
        pa.int64(): _to_pyint,
        pa.uint8(): _to_pyint,
        pa.uint16(): _to_pyint,
        pa.uint32(): _to_pyint,
        pa.uint64(): _to_pyint,
        pa.float16(): _to_pyfloat,
        pa.float32(): _to_pyfloat,
        pa.float64(): _to_pyfloat,
        pa.date32(): _to_pydate,
    }

    def __init__(
        self,
        schema: pa.Schema,
        copy: bool = True,
        deep: bool = False,
        str_as_json: bool = True,
    ):
        self._copy = copy
        self._deep = deep
        self._str_as_json = str_as_json

        self._to_pytype = [self._build_field_converter(f) for f in schema]

    def row_to_py(self, data: List[Any]) -> List[Any]:
        if not self._copy:
            for i in range(len(data)):
                data[i] = self._to_pytype[i](data[i])
            return data
        else:
            return [self._to_pytype[i](data[i]) for i in range(len(data))]

    def _build_field_converter(self, f: pa.Field) -> Callable[[Any], Any]:
        if f.type in _TypeConverter._CONVERTERS:
            return _TypeConverter._CONVERTERS[f.type]
        elif isinstance(f.type, pa.TimestampType):
            return _to_pydatetime
        elif isinstance(f.type, pa.Decimal128Type):
            raise NotImplementedError("decimal conversion is not supported")
        elif isinstance(f.type, pa.StructType):
            if not self._deep:
                return lambda x: _assert_pytype(dict, x)
            else:
                converters = {
                    x.name: self._build_field_converter(x) for x in list(f.type)
                }
                return lambda x: _to_pydict(converters, x, self._str_as_json)
        elif isinstance(f.type, pa.ListType):
            if not self._deep:
                return lambda x: _assert_pytype(list, x)
            else:
                converter = self._build_field_converter(
                    pa.field("e", f.type.value_type)
                )
                return lambda x: _to_pylist(converter, x, self._copy, self._str_as_json)
        raise NotImplementedError(f"{f} type is not supported")  # pragma: no cover
