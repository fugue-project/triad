from typing import Any, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np
from triad.collections.dataframe.dataframe import DataFrame, LocalDataFrame
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none, assert_or_throw


class PandasDataFrame(LocalDataFrame):
    def __init__(  # noqa: C901
        self, df: Any = None, schema: Any = None, metadata: Any = None
    ):
        if df is None:
            assert_arg_not_none(
                "schema", "Schema must be set if dataframe is not provided"
            )
            schema = Schema(schema)
            pdf = pd.DataFrame([], columns=schema.names)
            pdf = pdf.astype(dtype=schema.pd_dtype)
        elif isinstance(df, PandasDataFrame):
            pdf = df.native
            schema = None
        elif isinstance(df, (pd.DataFrame, pd.Series)):
            if isinstance(df, pd.Series):
                df = df.to_frame()
            pdf = df
            schema = None if schema is None else Schema(schema)
        elif isinstance(df, Iterable):
            assert_arg_not_none(schema, msg=f"schema can't be None for iterable input")
            schema = Schema(schema)
            pdf = pd.DataFrame(df, columns=schema.names)
            pdf = pdf.astype(dtype=schema.pd_dtype)
        else:
            raise ValueError(f"{df} is incompatible with PandasDataFrame")
        pdf, schema = self._apply_schema(pdf, schema)
        super().__init__(schema, metadata)
        self._native = pdf

    @property
    def native(self) -> pd.DataFrame:
        return self._native

    def empty(self) -> bool:
        return self.native.empty

    def peek_array(self) -> Any:
        return self.native.iloc[0].values.tolist()

    def count(self, persist: bool = False) -> int:
        return self.native.shape[0]

    def as_pandas(self) -> pd.DataFrame:
        return self._native

    def drop(self, cols: List[str]) -> DataFrame:
        try:
            schema = self.schema - cols
        except Exception as e:
            raise InvalidOperationError(str(e))
        if len(schema) == 0:
            raise InvalidOperationError("Can't remove all columns of a dataframe")
        return PandasDataFrame(self.native.drop(cols, axis=1), schema)

    def as_array(
        self, columns: Optional[List[str]] = None, type_safe: bool = False
    ) -> List[Any]:
        return list(self.as_array_iterable(columns))

    def as_array_iterable(
        self, columns: Optional[List[str]] = None, type_safe: bool = False
    ) -> Iterable[Any]:
        if self._native.shape[0] == 0:
            return
        if columns is None:
            sub = self.schema
        else:
            sub = self.schema.extract(columns)
        df = self._native[sub.names]
        for arr in self._as_array_iterable(df):
            yield arr
            # yield [self.schema.types[i].to_native(arr[i]) for i in rg]

    # ref: https://stackoverflow.com/questions/34838378/dataframe-values-tolist-datatype
    # stupid numpy casting, but if there is a string col, num type casting won't happen
    # TODO: there is a concern without this method, is it possible
    # during to list 1 -> 0.99999999, then during to_native 0.99999999 -> 0?
    def _as_array_iterable(self, df: pd.DataFrame) -> Iterable[Any]:
        if any(not np.issubdtype(t, np.number) for t in df.dtypes):
            for _, r in df.iterrows():
                yield r.tolist()
        else:
            temp_df = df.copy()
            temp_df["__deal_with_stupid_typing__"] = ""
            for _, r in temp_df.iterrows():
                yield r.tolist()[:-1]

    def _apply_schema(
        self, pdf: pd.DataFrame, schema: Optional[Schema]
    ) -> Tuple[pd.DataFrame, Schema]:
        assert_or_throw(
            pdf.empty or type(pdf.index) == pd.RangeIndex,
            ValueError("Pandas datafame must have default index"),
        )
        assert schema is None or len(schema) > 0, "Schema if set then it can't be empty"
        if pdf.columns.dtype == "object":  # pdf has named schema
            pschema = Schema(pdf)
            if schema is None or pschema == schema:
                return pdf, pschema
            pdf = pdf[schema.names]
        else:  # pdf has no named schema
            assert_arg_not_none(
                schema, msg="Schema can't be none when dataframe has no named schema"
            )
            assert schema is not None
            assert_or_throw(
                pdf.shape[1] == len(schema),
                ValueError(f"Pandas datafame column count doesn't match {schema}"),
            )
            pdf.columns = schema.names
        pdf = pdf.astype(dtype=schema.pd_dtype)
        return pdf, schema
