import json
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Iterable

import pandas as pd
import pyarrow as pa
from triad.utils.convert import as_type

# from triad.utils.iter import make_empty_aware
from triad.collections.schema import Schema


def apply_schema(
    schema: Schema,
    data: Iterable[List[Any]],
    copy: bool = True,
    deep: bool = False,
    str_as_json: bool = True,
) -> Iterable[List[Any]]:
    converter = _TypeConverter(schema, copy=copy, deep=deep, str_as_json=str_as_json)
    try:
        for item in data:
            yield converter.row_to_py(item)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(str(e))


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
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    if isinstance(obj, datetime):
        return obj
    return as_type(obj, datetime)


def _to_pydate(obj: Any) -> Any:
    if obj is None or obj is pd.NaT:
        return None
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
        schema: Schema,
        copy: bool = True,
        deep: bool = False,
        str_as_json: bool = True,
    ):
        self._copy = copy
        self._deep = deep
        self._str_as_json = str_as_json

        self._to_pytype = [
            self._build_field_converter(schema.get_value_by_index(i))
            for i in range(len(schema))
        ]

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
