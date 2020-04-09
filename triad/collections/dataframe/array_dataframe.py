from typing import Any, Iterable, List, Optional

from triad.collections.dataframe.dataframe import (
    DataFrame,
    LocalDataFrame,
    _get_schema_change,
)
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pyarrow import apply_schema


class ArrayDataFrame(LocalDataFrame):
    def __init__(  # noqa: C901
        self, df: Any = None, schema: Any = None, metadata: Any = None
    ):
        if df is None:
            assert_arg_not_none(
                schema, msg="Schema must be set if dataframe is not provided"
            )
            super().__init__(schema, metadata)
            self._native = []
        elif isinstance(df, DataFrame):
            if schema is None:
                super().__init__(df.schema, metadata)
                self._native = df.as_array(type_safe=False)
            else:
                schema, _ = _get_schema_change(df.schema, schema)
                super().__init__(schema, metadata)
                self._native = df.as_array(schema.names, type_safe=False)
        elif isinstance(df, Iterable):
            assert_arg_not_none(schema, msg="schema can't be None for iterable input")
            super().__init__(schema, metadata)
            self._native = df if isinstance(df, List) else list(df)
        else:
            raise ValueError(f"{df} is incompatible with ArrayDataFrame")

    @property
    def native(self) -> List[Any]:
        return self._native

    def empty(self) -> bool:
        return self.count() == 0

    def peek_array(self) -> Any:
        return list(self.native[0])

    def count(self, persist: bool = False) -> int:
        return len(self.native)

    def drop(self, cols: List[str]) -> DataFrame:
        try:
            schema = self.schema - cols
        except Exception as e:
            raise InvalidOperationError(str(e))
        if len(schema) == 0:
            raise InvalidOperationError("Can't remove all columns of a dataframe")
        return ArrayDataFrame(self, schema)

    def as_array(
        self, columns: Optional[List[str]] = None, type_safe: bool = False
    ) -> List[Any]:
        if not type_safe and columns is None:
            return self.native
        return list(self.as_array_iterable(columns, type_safe=type_safe))

    def as_array_iterable(
        self, columns: Optional[List[str]] = None, type_safe: bool = False
    ) -> Iterable[Any]:
        if columns is None:
            pos = []
        else:
            pos = [self.schema.index_of_key(k) for k in columns]
            assert_or_throw(len(pos) > 0, "columns if set must be non empty")
        if not type_safe:
            for item in self._iter_cols(pos):
                yield item
        else:
            sub = self.schema if columns is None else self.schema.extract(columns)
            for item in apply_schema(
                sub.pa_schema,
                self._iter_cols(pos),
                copy=True,
                deep=True,
                str_as_json=True,
            ):
                yield item

    def _iter_cols(self, pos: List[int]) -> Iterable[List[Any]]:
        if len(pos) == 0:
            for row in self.native:
                yield row
        else:
            for row in self.native:
                yield [row[p] for p in pos]