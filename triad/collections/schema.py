from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import pandas as pd
import pyarrow as pa
from triad.collections.dict import IndexedOrderedDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pyarrow import (
    _parse_type,
    expression_to_schema,
    schema_to_expression,
    validate_column_name,
)


class SchemaException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class Schema(IndexedOrderedDict):
    def __init__(self, *args: Any):
        super().__init__()
        self._append(list(args))

    @property
    def names(self) -> List[str]:
        self._build_index()
        return self._index_key  # type: ignore

    @property
    def fields(self) -> List[pa.Field]:
        return list(self.values())

    @property
    def types(self) -> List[pa.DataType]:
        return [v.value_type for v in self.values()]

    @property
    def pyarrow_schema(self) -> pa.Schema:
        return pa.schema(self.fields)

    def copy(self) -> "Schema":
        other = super().copy()
        assert isinstance(other, Schema)
        return other

    def __repr__(self) -> str:
        return schema_to_expression(self.pyarrow_schema)

    def __iadd__(self, obj: Any) -> "Schema":
        return self._append(obj)

    def __add__(self, obj: Any) -> "Schema":
        return self.copy()._append(obj)

    def __sub__(self, obj: Any) -> "Schema":
        return self._remove(obj, require_type_match=True, ignore_type_mismatch=False)

    def __setitem__(  # type: ignore
        self, name: str, value: Any, *args: List[Any], **kwds: Dict[str, Any]
    ) -> None:
        assert_arg_not_none(value, "value")
        if not validate_column_name(name):
            raise SchemaException(f"Invalid column name {name}")
        if name in self:  # update existing value is not allowed
            raise SchemaException(f"{name} already exists in {self}")
        if isinstance(value, pa.Field):
            assert_or_throw(
                name == value.name, SchemaException(f"{name} doesn't match {value}")
            )
        elif isinstance(value, pa.DataType):
            value = pa.field(name, value)
        elif isinstance(value, str):
            value = pa.field(name, _parse_type(value))
        super().__setitem__(name, value, *args, **kwds)  # type: ignore

    def __eq__(self, other: Any) -> bool:
        if other is None:
            return False
        if isinstance(other, Schema):
            return super().__eq__(other)
        if isinstance(other, str):
            return self.__repr__() == other
        if isinstance(other, (pa.Schema, pa.Field, pd.DataFrame, OrderedDict, List)):
            return self == Schema(other)
        else:
            raise Exception(f"what is this {other}")
        return False

    def _append(self, obj: Any) -> "Schema":  # noqa: C901
        if obj is None:
            return self
        elif isinstance(obj, pa.Field):
            self[obj.name] = obj.type
        elif isinstance(obj, str):
            self._append_expression(obj)
        elif isinstance(obj, Dict):
            for k, v in obj.items():
                self[k] = v
        elif isinstance(obj, pa.Schema):
            self._append_pa_schema(obj)
        elif isinstance(obj, pd.DataFrame):
            self._append_pd_df_schema(obj)
        elif isinstance(obj, tuple):
            self[obj[0]] = obj[1]
        elif isinstance(obj, List):
            for x in obj:
                self._append(x)
        else:
            raise ValueError(f"Invalid schema to add {obj}")
        return self

    def _remove(
        self,
        obj: Any,
        require_type_match: bool = True,
        ignore_type_mismatch: bool = False,
    ) -> "Schema":
        if obj is None:
            return self.copy()
        if isinstance(obj, str):
            if ":" in obj:  # expression
                ps = expression_to_schema(obj)
                pairs: List[Tuple[str, pa.DataType]] = list(zip(ps.names, ps.types))
            else:
                pairs = [(obj, None)]  # single key
        elif isinstance(obj, (pa.Schema, Schema)):
            pairs = list(zip(ps.names, ps.types))
        else:
            return self._remove(Schema(obj), require_type_match, ignore_type_mismatch)
        od = OrderedDict(self)
        for k, v in pairs:
            if v is None:
                del od[k]
            else:
                tp = od[k].type
                if tp == v:
                    del od[k]
                elif not ignore_type_mismatch:
                    raise SchemaException(
                        f"Unable to remove {k}:{v} from {self}, type mismatch"
                    )
        return Schema(od)

    def _pre_update(self, op: str, need_reindex: bool = True) -> None:
        if op == "__setitem__":
            super()._pre_update(op, need_reindex)
        else:
            raise InvalidOperationError(f"{op} is not allowed in Schema")

    def _append_pa_schema(self, other: pa.Schema) -> "Schema":
        for k, v in zip(other.names, other.types):
            self[k] = v
        return self

    def _append_expression(self, other: str) -> "Schema":
        o = expression_to_schema(other)
        return self._append_pa_schema(o)

    def _append_pd_df_schema(self, df: pd.DataFrame) -> "Schema":
        o = pa.Schema.from_pandas(df)
        return self._append_pa_schema(o)
