import json
import pickle
from datetime import date, datetime
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import io
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from pandas.core.dtypes.base import ExtensionDtype
from pyarrow.compute import CastOptions, binary_join_element_wise
from pyarrow.json import read_json, ParseOptions as JsonParseOptions
from triad.constants import TRIAD_VAR_QUOTE

from .convert import as_type
from .iter import EmptyAwareIterable, Slicer
from .json import loads_no_dup
from .schema import move_to_unquoted, quote_name, unquote_name
from .assertion import assert_or_throw

PYARROW_VERSION = version.parse(pa.__version__)
TRIAD_DEFAULT_TIMESTAMP_UNIT = "us"
TRIAD_DEFAULT_TIMESTAMP = pa.timestamp(TRIAD_DEFAULT_TIMESTAMP_UNIT)

LARGE_TYPES_REPLACEMENT: List[Tuple[Any, Any]] = [
    (pa.large_string(), pa.string()),
    (pa.large_binary(), pa.binary()),
    (pa.large_utf8(), pa.utf8()),
    (pa.types.is_large_list, lambda t: pa.list_(t.value_field)),
]


_TYPE_EXPRESSION_MAPPING: Dict[str, pa.DataType] = {
    "null": pa.null(),
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
    "datetime": TRIAD_DEFAULT_TIMESTAMP,
    "binary": pa.binary(),
    "bytes": pa.binary(),
}

_TYPE_EXPRESSION_R_MAPPING: Dict[pa.DataType, str] = {
    pa.null(): "null",
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
    TRIAD_DEFAULT_TIMESTAMP: "datetime",
    pa.binary(): "bytes",
}

_PANDAS_EXTENSION_TYPE_TO_PA_MAP: Dict[ExtensionDtype, pa.DataType] = {
    pd.Int8Dtype(): pa.int8(),
    pd.UInt8Dtype(): pa.uint8(),
    pd.Int16Dtype(): pa.int16(),
    pd.UInt16Dtype(): pa.uint16(),
    pd.Int32Dtype(): pa.int32(),
    pd.UInt32Dtype(): pa.uint32(),
    pd.Int64Dtype(): pa.int64(),
    pd.UInt64Dtype(): pa.uint64(),
    pd.Float32Dtype(): pa.float32(),
    pd.Float64Dtype(): pa.float64(),
    pd.StringDtype(): pa.string(),
    pd.BooleanDtype(): pa.bool_(),
}

_PA_TO_PANDAS_EXTENSION_TYPE_MAP: Dict[pa.DataType, ExtensionDtype] = {
    pa.int8(): pd.Int8Dtype(),
    pa.uint8(): pd.UInt8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.uint16(): pd.UInt16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.uint32(): pd.UInt32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.uint64(): pd.UInt64Dtype(),
    pa.float32(): pd.Float32Dtype(),
    pa.float64(): pd.Float64Dtype(),
    pa.string(): pd.StringDtype(),
    pa.bool_(): pd.BooleanDtype(),
}

_SPECIAL_TOKENS: Set[str] = {",", "{", "}", "[", "]", "<", ">", ":"}


def expression_to_schema(expr: str) -> pa.Schema:
    """Convert schema expression to `pyarrow.Schema`.

    Format: `col_name:col_type[,col_name:col_type]+`

    If col_type is a list type, the syntax should be `[element_type]`

    If col_type is a struct type, the syntax should
    be `{col_name:col_type[,col_name:col_type]+}`

    If col_type is a map type, the syntax should
    be `<key_type,value_type>`


    Whitespaces will be removed. The format of the expression is json
    without any double quotes

    .. admonition:: Examples

        .. code-block:: python

            expression_to_schema("a:int,b:int")
            expression_to_schema("a:[int],b:{x:<int,int>,y:{z:[str],w:byte}}")

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
        raise SyntaxError(f"Invalid syntax: '{expr}' {str(e)}")  # noqa


def schema_to_expression(schema: pa.Schema) -> pa.Schema:
    """Convert pyarrow.Schema to Triad schema expression
    see :func:`~triad.utils.pyarrow.expression_to_schema`

    :param schema: pyarrow schema
    :raises NotImplementedError: if there some type is not supported by Triad
    :return: schema string expression
    """
    return ",".join(_field_to_expression(x) for x in list(schema))


def to_pa_datatype(obj: Any) -> pa.DataType:  # noqa: C901
    """Convert an object to pyarrow DataType

    :param obj: any object
    :raises TypeError: if unable to convert
    :return: an instance of pd.DataType
    """
    if obj is None:
        raise TypeError("obj can't be None")
    if isinstance(obj, pa.DataType):
        return obj
    if obj is bool:
        return pa.bool_()
    if obj is int:
        return pa.int64()
    if obj is float:
        return pa.float64()
    if obj is str:
        return pa.string()
    if isinstance(obj, str):
        return _parse_type(obj)
    if isinstance(obj, ExtensionDtype):
        if obj in _PANDAS_EXTENSION_TYPE_TO_PA_MAP:
            return _PANDAS_EXTENSION_TYPE_TO_PA_MAP[obj]
        if hasattr(pd, "ArrowDtype"):
            if isinstance(obj, pd.ArrowDtype):
                return obj.pyarrow_dtype
            if obj == pd.StringDtype("pyarrow"):
                return pa.string()
    if type(obj) == type and issubclass(obj, datetime):
        return TRIAD_DEFAULT_TIMESTAMP
    if type(obj) == type and issubclass(obj, date):
        return pa.date32()
    return pa.from_numpy_dtype(np.dtype(obj))


def pa_datatypes_equal(  # noqa: C901
    t1: pa.DataType,
    t2: pa.DataType,
    ignore_list_item_name: bool = True,
    equal_groups: Optional[List[List[Callable[[pa.DataType], bool]]]] = None,
) -> bool:
    """Check if two pyarrow data types are equal

    :param t1: the first pyarrow data type
    :param t2: the second pyarrow data type
    :param ignore_list_item_name: whether to ignore list item name,
        defaults to True
    :param equal_groups: a list of groups of functions to check equality,
        defaults to None

    :return: if the two data types are equal

    .. note::

        In the lastest version of pyarrow, in the default comparison logic,
        list field names are not compared.

    .. admonition:: Examples

        .. code-block:: python

            assert not pa_datatypes_equal(pa.int32(), pa.int64())
            assert pa_datatypes_equal(
                pa.int32(),
                pa.int64(),
                equal_groups=[[pa.types.is_integer]],
            )
    """
    if t1 is t2:
        return True
    if (
        not ignore_list_item_name
        and pa.types.is_list(t1)
        and pa.types.is_list(t2)
        and t1.value_field.name != t2.value_field.name
    ):
        return False
    if t1 == t2:
        return True
    if equal_groups is not None:
        for group in equal_groups:
            if any(f(t1) for f in group) and any(f(t2) for f in group):
                return True
    params: Dict[str, Any] = dict(  # noqa: C408
        ignore_list_item_name=ignore_list_item_name,
        equal_groups=equal_groups,
    )
    if pa.types.is_list(t1) and pa.types.is_list(t2):  # pragma: no cover
        # for lower version of pyarrow, list field names are compared
        # for higher version of pyarrow, list field names are ignored
        return pa_datatypes_equal(t1.value_type, t2.value_type, **params)
    if pa.types.is_struct(t1) and pa.types.is_struct(t2):
        if len(t1) != len(t2):
            return False
        for f1, f2 in zip(t1, t2):
            if f1.name != f2.name:
                return False
            if not pa_datatypes_equal(f1.type, f2.type, **params):
                return False
        return True
    if pa.types.is_map(t1) and pa.types.is_map(t2):
        return pa_datatypes_equal(
            t1.key_type, t2.key_type, **params
        ) and pa_datatypes_equal(t1.item_type, t2.item_type, **params)

    return False


def pa_schemas_equal(
    s1: pa.Schema,
    s2: pa.Schema,
    ignore_list_item_name: bool = True,
    equal_groups: Optional[List[List[Callable[[pa.DataType], bool]]]] = None,
) -> bool:
    """Check if two pyarrow schemas are equal

    :param s1: the first pyarrow schema
    :param s2: the second pyarrow schema
    :param ignore_list_item_name: whether to ignore list item name,
        defaults to True
    :param equal_groups: a list of groups of functions to check equality,
        defaults to None

    :return: if the two schemas are equal

    .. note::

        In the lastest version of pyarrow, in the default comparison logic,
        list field names are not compared.

    .. admonition:: Examples

        .. code-block:: python

            s1 = pa.schema([("a", pa.int32()), ("b", pa.string())])
            s2 = pa.schema([("a", pa.int64()), ("b", pa.string())])
            assert not pa_schemas_equal(s1, s2)
            assert pa_schemas_equal(
                s1,
                s2,
                equal_groups=[[pa.types.is_integer]],
            )
    """
    if ignore_list_item_name:
        if s1 is s2 or s1.equals(s2):
            return True
    elif s1 is s2:
        return True
    if s1.names != s2.names:
        return False
    for f1, f2 in zip(s1, s2):
        if not pa_datatypes_equal(
            f1.type,
            f2.type,
            ignore_list_item_name=ignore_list_item_name,
            equal_groups=equal_groups,
        ):
            return False
    return True


def cast_pa_array(col: pa.Array, new_type: pa.DataType) -> pa.Array:  # noqa: C901
    old_type = col.type
    if new_type.equals(old_type):
        return col
    elif pa.types.is_date(new_type) and not pa.types.is_date(old_type):
        # -> date
        return pa.Array.from_pandas(pd.to_datetime(col.to_pandas()).dt.date)
    elif pa.types.is_timestamp(new_type):
        # -> datetime
        if pa.types.is_timestamp(old_type) or pa.types.is_date(old_type):
            s = pd.to_datetime(col.to_pandas())
            from_tz = old_type.tz if pa.types.is_timestamp(old_type) else None
            to_tz = new_type.tz
            if from_tz is None or to_tz is None:
                s = s.dt.tz_localize(to_tz)
            else:
                s = s.dt.tz_convert(to_tz)
        else:
            s = pd.to_datetime(col.to_pandas())
        return pa.Array.from_pandas(s, type=new_type)
    elif pa.types.is_integer(new_type):
        if PYARROW_VERSION.major < 9:  # pragma: no cover
            return col.cast(new_type, safe=False)
        return col.cast(
            options=CastOptions(
                new_type, allow_decimal_truncate=True, allow_float_truncate=True
            ),
        )
    elif pa.types.is_string(new_type):
        if pa.types.is_timestamp(old_type):
            # datetime -> str
            # this is to ensure less granular ts series won't output .000000
            series = pd.to_datetime(col.to_pandas())
            ns = series.isnull()
            series = series.astype(str)
            return pa.Array.from_pandas(series.mask(ns, None), type=new_type)
    return col.cast(new_type, safe=True)


def cast_pa_table(df: pa.Table, schema: pa.Schema) -> pa.Table:
    """Convert a pyarrow table to another pyarrow table with given schema

    :param df: the pyarrow table
    :param schema: the pyarrow schema
    :return: the converted pyarrow table
    """
    if df.schema == schema:
        return df
    cols = [cast_pa_array(col, new_f.type) for col, new_f in zip(df.columns, schema)]
    return pa.Table.from_arrays(cols, schema=schema)


def to_pandas_types_mapper(
    pa_type: pa.DataType,
    use_extension_types: bool = False,
    use_arrow_dtype: bool = False,
) -> Optional[pd.api.extensions.ExtensionDtype]:
    """The types_mapper for ``pa.Table.to_pandas``

    :param pa_type: the pyarrow data type
    :param use_extension_types: whether to use pandas extension
        data types, default to False
    :param use_arrow_dtype: if True and when pandas supports ``ArrowDType``,
        use pyarrow types, default False
    :return: the pandas ExtensionDtype if available, otherwise None

    .. note::

        * If ``use_extension_types`` is False and ``use_arrow_dtype`` is True,
            it converts the type to ``ArrowDType``
        * If both are true, it converts the type to the numpy backend nullable
            dtypes if possible, otherwise, it converts to ``ArrowDType``
    """
    use_arrow_dtype = use_arrow_dtype and hasattr(pd, "ArrowDtype")
    if use_extension_types:
        return (
            _PA_TO_PANDAS_EXTENSION_TYPE_MAP[pa_type]
            if pa_type in _PA_TO_PANDAS_EXTENSION_TYPE_MAP
            else (None if not use_arrow_dtype else pd.ArrowDtype(pa_type))
        )
    if use_arrow_dtype:
        return pd.ArrowDtype(pa_type)
    return None


def pa_table_to_pandas(
    df: pa.Table,
    use_extension_types: bool = False,
    use_arrow_dtype: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convert a pyarrow table to pandas dataframe

    :param df: the pyarrow table
    :param use_extension_types: whether to use pandas extension
        data types, default to False
    :param use_arrow_dtype: if True and when pandas supports ``ArrowDType``,
        use pyarrow types, default False
    :param kwargs: other arguments for ``pa.Table.to_pandas``
    :return: the pandas dataframe
    """

    def _get_batches() -> Iterable[pa.RecordBatch]:
        if df.num_rows == 0:
            yield pa.RecordBatch.from_pydict(
                {name: [] for name in df.schema.names}, schema=df.schema
            )
        else:
            for batch in df.to_batches():
                if batch.num_rows > 0:
                    yield batch

    return pd.concat(
        pa_batch_to_pandas(batch, use_extension_types, use_arrow_dtype, **kwargs)
        for batch in _get_batches()
    )


def pa_batch_to_pandas(
    batch: pa.RecordBatch,
    use_extension_types: bool = False,
    use_arrow_dtype: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convert a pyarrow record batch to pandas dataframe

    :param batch: the pyarrow record batch
    :param use_extension_types: whether to use pandas extension
        data types, default to False
    :param use_arrow_dtype: if True and when pandas supports ``ArrowDType``,
        use pyarrow types, default False
    :param kwargs: other arguments for ``pa.Table.to_pandas``
    :return: the pandas dataframe
    """
    use_arrow_dtype = use_arrow_dtype and hasattr(pd, "ArrowDtype")
    mapper = partial(
        to_pandas_types_mapper,
        use_extension_types=use_extension_types,
        use_arrow_dtype=use_arrow_dtype,
    )
    return batch.to_pandas(types_mapper=mapper, **kwargs)


def pa_batch_to_dicts(batch: pa.RecordBatch) -> List[Dict[str, Any]]:
    """Convert a pyarrow record batch to list of dict

    :param batch: the pyarrow record batch
    :return: the list of dict
    """
    if PYARROW_VERSION.major < 7:  # pragma: no cover
        names = batch.schema.names
        return [dict(zip(names, tp)) for tp in zip(*batch.to_pydict().values())]
    else:
        return batch.to_pylist()


def to_single_pandas_dtype(
    pa_type: pa.DataType,
    use_extension_types: bool = False,
    use_arrow_dtype: bool = False,
) -> np.dtype:
    """convert a pyarrow data type to a pandas datatype.
    Currently, struct type is not supported

    :param pa_type: the pyarrow data type
    :param use_extension_types: whether to use pandas extension
        data types, default to False
    :param use_arrow_dtype: if True and when pandas supports ``ArrowDType``,
        use pyarrow types, default False
    :return: the pandas data type

    .. note::

        * If ``use_extension_types`` is False and ``use_arrow_dtype`` is True,
            it converts the type to ``ArrowDType``
        * If both are true, it converts the type to the numpy backend nullable
            dtypes if possible, otherwise, it converts to ``ArrowDType``
    """
    use_arrow_dtype = use_arrow_dtype and hasattr(pd, "ArrowDtype")
    if pa.types.is_nested(pa_type) and not use_arrow_dtype:
        return np.dtype(object)
    tp = to_pandas_types_mapper(
        pa_type,
        use_extension_types=use_extension_types,
        use_arrow_dtype=use_arrow_dtype,
    )
    if tp is not None:
        return tp
    if pa.types.is_string(pa_type) and not use_extension_types and not use_arrow_dtype:
        return np.dtype(str)
    return pa_type.to_pandas_dtype()


def to_pandas_dtype(
    schema: pa.Schema,
    use_extension_types: bool = False,
    use_arrow_dtype: bool = False,
) -> Dict[str, np.dtype]:
    """convert as `dtype` dict for pandas dataframes.
    Currently, struct type is not supported

    :param schema: the pyarrow schema
    :param use_extension_types: whether to use pandas extension
        data types, default to False
    :param use_arrow_dtype: if True and when pandas supports ``ArrowDType``,
        use pyarrow types, default False
    :return: the pandas data type dictionary

    .. note::

        * If ``use_extension_types`` is False and ``use_arrow_dtype`` is True,
            it converts all types to ``ArrowDType``
        * If both are true, it converts types to the numpy backend nullable
            dtypes if possible, otherwise, it converts to ``ArrowDType``
    """
    return {
        f.name: to_single_pandas_dtype(
            f.type,
            use_extension_types=use_extension_types,
            use_arrow_dtype=use_arrow_dtype,
        )
        for f in schema
    }


def is_supported(data_type: pa.DataType, throw: bool = False) -> bool:
    """Whether `data_type` is currently supported by Triad

    :param data_type: instance of pa.DataType
    :param throw: whether to raise exception if not supported
    :return: if it is supported
    """
    if data_type in _TYPE_EXPRESSION_R_MAPPING:
        return True
    tp = (pa.Decimal128Type, pa.TimestampType, pa.StructType, pa.ListType, pa.MapType)
    if isinstance(data_type, tp):
        return True
    if not throw:
        return False
    raise NotImplementedError(f"{data_type} is not supported by Triad")


def get_alter_func(
    from_schema: pa.Schema, to_schema: pa.Schema, safe: bool
) -> Callable[[pa.Table], pa.Table]:
    """Generate the alteration function based on ``from_schema`` and
    ``to_schema``. This function can be applied to arrow tables with
    ``from_schema``, the outout will be in ``to_schema``'s order and types

    :param from_schema: the source schema
    :param to_schema: the destination schema
    :param safe: whether to check for conversion errors such as overflow
    :return: a function that can be applied to arrow tables with
        ``from_schema``, the outout will be in ``to_schema``'s order
        and types
    """
    params: List[Tuple[pa.Field, int, bool]] = []
    same = True
    for i, f in enumerate(to_schema):
        j = from_schema.get_field_index(f.name)
        if j < 0:
            raise KeyError(f"{f.name} is not in {from_schema}")
        other = from_schema.field(j)
        pos_same, type_same = (i == j), (f.type == other.type)
        same &= pos_same & type_same
        params.append((f, j, type_same))

    def _alter(df: pa.Table, params: List[Tuple[pa.Field, int, bool]]) -> pa.Table:
        cols: List[pa.ChunkedArray] = []
        names: List[str] = []
        for field, pos, same in params:
            names.append(field.name)
            col = df.column(pos)
            if not same:
                col = col.cast(field.type, safe=safe)
            cols.append(col)

        return pa.Table.from_arrays(cols, names=names)

    if same:
        return lambda x: x
    return partial(_alter, params=params)


def replace_types_in_table(
    df: pa.Table,
    pairs: List[
        Tuple[
            Union[Callable[[pa.DataType], bool], pa.DataType],
            Union[Callable[[pa.DataType], pa.DataType], pa.DataType],
        ]
    ],
    recursive: bool = True,
    safe: bool = True,
) -> pa.Table:
    """Replace(cast) types in a table

    :param df: the table
    :param pairs: a list of (is_type, convert_type) pairs
    :param recursive: whether to do recursive replacement in nested types
    :param safe: whether to check for conversion errors such as overflow

    :return: the new table
    """
    old_schema = df.schema
    new_schema = replace_types_in_schema(old_schema, pairs, recursive)
    if old_schema is new_schema:
        return df
    func = get_alter_func(old_schema, new_schema, safe=safe)
    return func(df)


def replace_types_in_schema(
    schema: pa.Schema,
    pairs: List[
        Tuple[
            Union[Callable[[pa.DataType], bool], pa.DataType],
            Union[Callable[[pa.DataType], pa.DataType], pa.DataType],
        ]
    ],
    recursive: bool = True,
) -> pa.Schema:
    """Replace types in a schema

    :param schema: the schema
    :param pairs: a list of (is_type, convert_type) pairs
    :param recursive: whether to do recursive replacement in nested types
    :return: the new schema
    """
    fields = []
    changed = False
    for f in schema:
        new_type = f.type
        for is_type, convert_type in pairs:
            _is_type = is_type if callable(is_type) else lambda t: t == is_type  # noqa
            _convert_type = (
                convert_type
                if callable(convert_type)
                else lambda t: convert_type  # noqa
            )
            new_type = replace_type(
                new_type, _is_type, _convert_type, recursive=recursive
            )
        if f.type is new_type or f.type == new_type:
            fields.append(f)
        else:
            changed = True
            fields.append(pa.field(f.name, new_type))
    if not changed:
        return schema
    return pa.schema(fields)


def replace_type(  # noqa: C901
    current_type: pa.DataType,
    is_type: Callable[[pa.DataType], bool],
    convert_type: Callable[[pa.DataType], pa.DataType],
    recursive: bool = True,
) -> pa.DataType:
    """Replace ``current_type`` or if it is nested, replace in the nested
    types

    :param current_type: the current type
    :param is_type: the function to check if the type is the type to replace
    :param convert_type: the function to convert the type
    :param recursive: whether to do recursive replacement in nested types
    :return: the new type
    """
    if not pa.types.is_nested(current_type) and is_type(current_type):
        return convert_type(current_type)
    if recursive:
        if pa.types.is_struct(current_type):
            old_fields = list(current_type)
            fields = [
                _replace_field(f, is_type, convert_type, recursive=recursive)
                for f in old_fields
            ]
            if all(a is b for a, b in zip(fields, old_fields)):
                return current_type
            return pa.struct(fields)
        if pa.types.is_list(current_type) or pa.types.is_large_list(current_type):
            old_f = current_type.value_field
            f = _replace_field(old_f, is_type, convert_type, recursive=recursive)
            if f is old_f:
                res = current_type
            elif pa.types.is_large_list(current_type):
                res = pa.large_list(f)
            else:
                res = pa.list_(f)
            if is_type(res):
                return convert_type(res)
            return res
        if pa.types.is_map(current_type):
            old_k, old_v = current_type.key_field, current_type.item_field
            k, v = _replace_field(
                old_k, is_type, convert_type, recursive=recursive
            ), _replace_field(old_v, is_type, convert_type, recursive=recursive)
            if k is old_k and v is old_v:
                return current_type
            return pa.map_(k, v)
    return current_type


def parse_json_columns(df: pa.Table, columns: Union[List[str], pa.Schema]) -> pa.Table:
    """Parse json string columns in a table and replace them with
    pyarrow types.

    :param df: the table
    :param columns: the columns to convert, can be a list of column names
        or a schema. If it is a list of names, then their types will be
        inferred from the data.
    :return: the new table
    """
    if isinstance(columns, list):
        sub = df.select(columns)
        options: Any = None
    else:
        sub = df.select([f.name for f in columns])
        schema = pa.schema([pa.field(f"col{i}", f.type) for i, f in enumerate(columns)])
        options = JsonParseOptions(explicit_schema=schema)
    if sub.num_columns == 0:
        return df
    assert_or_throw(
        all(pa.types.is_string(tp) for tp in sub.schema.types),
        ValueError("all selected columns must be string"),
    )
    name_map: Dict[str, int] = {}
    args: List[Any] = []
    for i, name in enumerate(sub.column_names):
        key = f'"col{i}":'
        if i > 0:
            key = "," + key
        else:
            key = "{" + key
        args.append(key)
        args.append(sub.column(i))
        name_map[name] = i
        if i == len(sub.column_names) - 1:
            args.append("}")
    jcol = binary_join_element_wise(
        args[0], args[1], "", null_handling="replace", null_replacement="null"
    )
    for i in range(2, len(args)):
        jcol = binary_join_element_wise(
            jcol, args[i], "", null_handling="replace", null_replacement="null"
        )
    json_stream = io.BytesIO(str.encode("\n".join(jcol.to_pylist())))
    parsed = read_json(json_stream, parse_options=options)
    cols: List[Any] = []
    for name, col in zip(df.schema.names, df.columns):
        if name not in name_map:
            cols.append(col)
        else:
            cols.append(parsed.column(name_map[name]))
    return pa.Table.from_arrays(cols, names=df.column_names)


def _replace_field(
    field: pa.Field,
    is_type: Callable[[pa.DataType], bool],
    convert_type: Callable[[pa.DataType], pa.DataType],
    recursive: bool,
):
    old_type = field.type
    new_type = replace_type(old_type, is_type, convert_type, recursive=recursive)
    if old_type is new_type or old_type == new_type:
        return field
    return pa.field(
        field.name, new_type, nullable=field.nullable, metadata=field.metadata
    )


def schemas_equal(
    a: pa.Schema,
    b: pa.Schema,
    check_order: bool = True,
    check_metadata: bool = True,
    ignore: Optional[
        List[
            Tuple[
                Union[Callable[[pa.DataType], bool], pa.DataType],
                Union[Callable[[pa.DataType], pa.DataType], pa.DataType],
            ]
        ]
    ] = None,
) -> bool:
    """check if two schemas are equal

    :param a: first pyarrow schema
    :param b: second pyarrow schema
    :param compare_order: whether to compare order
    :param compare_order: whether to compare metadata
    :param ignore: a list of (is_type, convert_type) pairs to
        ignore differences on, defaults to None
    :return: if the two schema equal
    """
    if a is b:
        return True
    if ignore is not None:
        a = replace_types_in_schema(a, ignore, recursive=True)
        b = replace_types_in_schema(b, ignore, recursive=True)
    if check_order:
        return a.equals(b, check_metadata=check_metadata)
    if check_metadata and a.metadata != b.metadata:
        return False
    da = {k: a.field(k) for k in a.names}
    db = {k: b.field(k) for k in b.names}
    return da == db


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


def get_eq_func(data_type: pa.DataType) -> Callable[[Any, Any], bool]:
    """Generate equality function for a give datatype

    :param data_type: pyarrow data type supported by Triad
    :return: the function
    """
    is_supported(data_type, throw=True)
    if data_type in _COMPARATORS:
        return _COMPARATORS[data_type]
    if pa.types.is_date(data_type):
        return _date_eq
    if pa.types.is_timestamp(data_type):
        return _timestamp_eq
    return _general_eq


class SchemaedDataPartitioner:
    """Partitioner for stream of array like data with given schema.
    It uses :func"`~triad.utils.iter.Slicer` to partition the stream

    :param schema: the schema of the data stream to process
    :param key_positions: positions of partition keys on `schema`
    :param sizer: the function to get size of an item
    :param row_limit: max row for each slice, defaults to None
    :param size_limit: max byte size for each slice, defaults to None
    """

    def __init__(
        self,
        schema: pa.Schema,
        key_positions: List[int],
        sizer: Optional[Callable[[Any], int]] = None,
        row_limit: int = 0,
        size_limit: Any = None,
    ):
        self._eq_funcs: List[Any] = [None] * len(schema)
        self._keys = key_positions
        for p in key_positions:
            self._eq_funcs[p] = get_eq_func(schema.types[p])
        self._slicer = Slicer(
            sizer=sizer,
            row_limit=row_limit,
            size_limit=size_limit,
            slicer=self._is_boundary,
        )
        self._hitting_boundary = True

    def partition(
        self, data: Iterable[Any]
    ) -> Iterable[Tuple[int, int, EmptyAwareIterable[Any]]]:
        """Partition the given data stream

        :param data: iterable of array like objects
        :yield: iterable of <partition_no, slice_no, slice iterable> tuple
        """
        self._hitting_boundary = False
        slice_no = 0
        partition_no = 0
        for slice_ in self._slicer.slice(data):
            if self._hitting_boundary:
                slice_no = 0
                partition_no += 1
                self._hitting_boundary = False
            yield partition_no, slice_no, slice_
            slice_no += 1

    def _is_boundary(self, no: int, current: Any, last: Any) -> bool:
        self._hitting_boundary = any(
            not self._eq_funcs[p](current[p], last[p]) for p in self._keys
        )
        return self._hitting_boundary


def _field_to_expression(field: pa.Field) -> str:
    name = quote_name(field.name)
    return f"{name}:{_type_to_expression(field.type)}"


def _type_to_expression(dt: pa.DataType) -> str:
    if dt in _TYPE_EXPRESSION_R_MAPPING:
        return _TYPE_EXPRESSION_R_MAPPING[dt]
    if isinstance(dt, pa.TimestampType):
        if dt.tz is None:
            return "datetime"
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
    if isinstance(dt, pa.MapType):
        k = _type_to_expression(dt.key_type)
        v = _type_to_expression(dt.item_type)
        return "<" + k + "," + v + ">"
    raise NotImplementedError(f"{dt} is not supported")


def _parse_types(v: Any):
    if isinstance(v, str):
        return _parse_type(v)
    elif isinstance(v, Dict):
        return pa.struct(_construct_struct(v))
    elif isinstance(v, List):
        if len(v) == 1:
            return pa.list_(_parse_types(v[0]))
        elif len(v) == 3 and v[0] is None:
            return pa.map_(_parse_types(v[1]), _parse_types(v[2]))
        raise SyntaxError(f"{v} is neither a list type nor a map type")
    else:  # pragma: no cover
        raise SyntaxError(f"{v} is not a valid type")


def _construct_struct(obj: Dict[str, Any]) -> Iterable[pa.Field]:
    for k, v in obj.items():
        yield pa.field(k, _parse_types(v))


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
    assert name.isidentifier(), f"Invalid expression {expr}"
    if len(p) == 1:
        return name, []
    arg_expr = p[1].strip().rstrip(")")
    args = [x.strip() for x in arg_expr.split(",")]
    return name, args


def _parse_tokens(expr: str) -> Iterable[str]:
    # parse to tokens that can construct a valid json string
    expr += ","
    last = 0
    skip = False
    i = 0
    while i < len(expr):
        if expr[i] == TRIAD_VAR_QUOTE:
            e = move_to_unquoted(expr, i)
            s = unquote_name(expr[i:e])
            yield '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
            last = i = e
            continue
        if expr[i] == "(":
            skip = True
        if expr[i] == ")":
            skip = False
        if not skip and expr[i] in _SPECIAL_TOKENS:
            s = expr[last:i].strip()
            if s != "":
                yield '"' + s + '"'
            if expr[i] == "<":  # <str,int> => [null,"str","int"]
                yield "[null,"
            elif expr[i] == ">":
                yield "]"
            else:
                yield expr[i]
            last = i + 1
        i += 1


def _to_pynone(obj: Any) -> Any:
    return None


def _to_pyint(obj: Any) -> Any:
    if obj is None or obj != obj:  # NaN
        return None
    if isinstance(obj, int):
        return obj
    return as_type(obj, int)


def _to_pystr(obj: Any) -> Any:
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _to_pybool(obj: Any) -> Any:
    if obj is None or obj != obj:  # NaN
        return None
    if isinstance(obj, bool):
        return obj
    return as_type(obj, bool)


def _to_pyfloat(obj: Any) -> Any:
    if obj is None or obj != obj:  # NaN
        return None
    if isinstance(obj, float):
        return obj
    obj = as_type(obj, float)
    return None if obj != obj else obj


def _to_pydatetime(obj: Any) -> Any:
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    if isinstance(obj, datetime):
        return obj
    obj = as_type(obj, datetime)
    return None if obj != obj else obj


def _to_pydate(obj: Any) -> Any:
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime().date()
    if isinstance(obj, datetime):
        return obj.date()
    if isinstance(obj, date):
        return obj
    obj = as_type(obj, datetime).date()
    return None if obj != obj else obj


def _to_pybytes(obj: Any) -> Any:
    if obj is None or isinstance(obj, bytes):
        return obj
    if isinstance(obj, bytearray):
        return bytes(obj)
    return pickle.dumps(obj)


def _assert_pytype(pytype: Any, obj: Any) -> Any:
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
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if not isinstance(obj, List):
        raise TypeError(f"{obj} is not list")
    if copy:
        return [converter(x) for x in obj]
    else:
        for i in range(len(obj)):
            obj[i] = converter(obj[i])
        return obj


def _to_pymap(
    key_f: Callable, value_f: Callable, obj: Any, str_as_json: bool = True
) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str) and str_as_json:
        obj = json.loads(obj)
    if isinstance(obj, Dict):
        return [(key_f(k), value_f(v)) for k, v in obj.items()]
    if isinstance(obj, list):
        return [(key_f(k), value_f(v)) for k, v in obj]
    raise NotImplementedError(f"{obj} can't be converted to map")


def _no_op_convert(obj: Any) -> Any:  # pragma: no cover
    return obj


class _TypeConverter:
    _CONVERTERS: Dict[pa.DataType, Any] = {
        pa.null(): _to_pynone,
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
        pa.binary(): _to_pybytes,
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
        try:
            return self._build_type_converter(f.type)
        except Exception:  # pragma: no cover
            raise NotImplementedError(f"{f} type is not supported")

    def _build_type_converter(self, tp: pa.DataType) -> Callable[[Any], Any]:
        if tp in _TypeConverter._CONVERTERS:
            return _TypeConverter._CONVERTERS[tp]
        elif pa.types.is_timestamp(tp):
            return _to_pydatetime
        elif pa.types.is_decimal(tp):
            raise NotImplementedError("decimal conversion is not supported")
        elif pa.types.is_struct(tp):
            if not self._deep:
                return lambda x: _assert_pytype(dict, x)
            else:
                converters = {x.name: self._build_field_converter(x) for x in list(tp)}
                return lambda x: _to_pydict(converters, x, self._str_as_json)
        elif pa.types.is_list(tp):
            if not self._deep:
                return lambda x: _assert_pytype(list, x)
            else:
                converter = self._build_type_converter(tp.value_type)
                return lambda x: _to_pylist(converter, x, self._copy, self._str_as_json)
        elif pa.types.is_map(tp):
            if not self._deep:
                return lambda x: _assert_pytype((dict, list), x)
            else:
                k = self._build_type_converter(tp.key_type)
                v = self._build_type_converter(tp.item_type)
                return lambda x: _to_pymap(k, v, x, self._str_as_json)
        raise NotImplementedError  # pragma: no cover


def _none_eq(o1: Any, o2: Any) -> bool:
    return True


def _general_eq(o1: Any, o2: Any) -> bool:
    return o1 == o2


def _float_eq(o1: Any, o2: Any) -> bool:
    return o1 == o2 or ((o1 != o1 or o1 is None) and (o2 != o2 or o2 is None))


def _timestamp_eq(o1: Any, o2: Any) -> bool:
    return o1 == o2 or ((o1 is pd.NaT or o1 is None) and (o2 is pd.NaT or o2 is None))


def _date_eq(o1: Any, o2: Any) -> bool:
    if o1 == o2:
        return True
    nat1 = o1 is pd.NaT or o1 is None
    nat2 = o2 is pd.NaT or o2 is None
    if nat1 and nat2:
        return True
    if nat1 or nat2:
        return False
    return o1.year == o2.year and o1.month == o2.month and o1.day == o2.day


_COMPARATORS: Dict[pa.DataType, Callable[[Any, Any], bool]] = {
    pa.null(): _none_eq,
    pa.float16(): _float_eq,
    pa.float32(): _float_eq,
    pa.float64(): _float_eq,
}
