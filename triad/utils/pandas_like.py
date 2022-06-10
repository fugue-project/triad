from datetime import datetime
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar

import numpy as np
import pandas as pd
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
    TRIAD_DEFAULT_TIMESTAMP,
    apply_schema,
    to_pandas_dtype,
    to_single_pandas_dtype,
)

import pyarrow as pa

T = TypeVar("T", bound=Any)
_DEFAULT_JOIN_KEYS: List[str] = []
_DEFAULT_DATETIME = datetime(2000, 1, 1)


class PandasLikeUtils(Generic[T]):
    """A collection of utils for general pandas like dataframes"""

    def empty(self, df: T) -> bool:
        """Check if the dataframe is empty

        :param df: pandas like dataframe
        :return: if it is empty
        """
        return len(df.index) == 0

    def as_arrow(self, df: T, schema: Optional[pa.Schema] = None) -> pa.Table:
        """Convert pandas like dataframe to pyarrow table

        :param df: pandas like dataframe
        :param schema: if specified, it will be used to construct pyarrow table,
          defaults to None

        :return: pyarrow table
        """
        return pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)

    def as_array_iterable(
        self,
        df: T,
        schema: Optional[pa.Schema] = None,
        columns: Optional[List[str]] = None,
        type_safe: bool = False,
    ) -> Iterable[List[Any]]:
        """Convert pandas like dataframe to iterable of rows in the format of list.

        :param df: pandas like dataframe
        :param schema: schema of the input. With None, it will infer the schema,
          it can't infer wrong schema for nested types, so try to be explicit
        :param columns: columns to output, None for all columns
        :param type_safe: whether to enforce the types in schema, if False, it will
            return the original values from the dataframe
        :return: iterable of rows, each row is a list

        :Notice:
        If there are nested types in schema, the conversion can be slower
        """
        if self.empty(df):
            return
        if schema is None:
            schema = self.to_schema(df)
        if columns is not None:
            df = df[columns]
            schema = pa.schema([schema.field(n) for n in columns])
        if not type_safe:
            for arr in df.itertuples(index=False, name=None):
                yield list(arr)
        elif all(not pa.types.is_nested(x) for x in schema.types):
            p = self.as_arrow(df, schema)
            d = p.to_pydict()
            cols = [d[n] for n in schema.names]
            for arr in zip(*cols):
                yield list(arr)
        else:
            # If schema has nested types, the conversion will be much slower
            for arr in apply_schema(
                schema,
                df.itertuples(index=False, name=None),
                copy=True,
                deep=True,
                str_as_json=True,
            ):
                yield arr

    def to_schema(self, df: T) -> pa.Schema:
        """Extract pandas dataframe schema as pyarrow schema. This is a replacement
        of pyarrow.Schema.from_pandas, and it can correctly handle string type and
        empty dataframes

        :param df: pandas dataframe
        :raises ValueError: if pandas dataframe does not have named schema
        :return: pyarrow.Schema

        :Notice:
        The dataframe must be either empty, or with type pd.RangeIndex, pd.Int64Index
        or pd.UInt64Index and without a name, otherwise, `ValueError` will raise.
        """
        self.ensure_compatible(df)
        assert_or_throw(
            df.columns.dtype == "object",
            ValueError("Pandas dataframe must have named schema"),
        )

        def get_fields() -> Iterable[pa.Field]:
            if isinstance(df, pd.DataFrame) and len(df.index) > 0:
                yield from pa.Schema.from_pandas(df, preserve_index=False)
            else:
                for i in range(df.shape[1]):
                    tp = df.dtypes[i]
                    if tp == np.dtype("object") or tp == np.dtype(str):
                        t = pa.string()
                    else:
                        t = pa.from_numpy_dtype(tp)
                    yield pa.field(df.columns[i], t)

        fields: List[pa.Field] = []
        for field in get_fields():
            if pa.types.is_timestamp(field.type):
                fields.append(pa.field(field.name, TRIAD_DEFAULT_TIMESTAMP))
            else:
                fields.append(field)
        return pa.schema(fields)

    def enforce_type(  # noqa: C901
        self, df: T, schema: pa.Schema, null_safe: bool = False
    ) -> T:
        """Enforce the pandas like dataframe to comply with `schema`.

        :param df: pandas like dataframe
        :param schema: pyarrow schema
        :param null_safe: whether to enforce None value for int, string and bool values
        :return: converted dataframe

        :Notice:
        When `null_safe` is true, the native column types in the dataframe may change,
        for example, if a column of `int64` has None values, the output will make sure
        each value in the column is either None or an integer, however, due to the
        behavior of pandas like dataframes, the type of the columns may
        no longer be `int64`

        This method does not enforce struct and list types
        """
        if self.empty(df):
            return df
        if not null_safe:
            return df.astype(dtype=to_pandas_dtype(schema, use_extension_types=False))
        data: Dict[str, Any] = {}
        for v in schema:
            s = df[v.name]
            if pa.types.is_string(v.type):
                ns = s.isnull()
                s = s.astype(str).mask(ns, None)
            elif pa.types.is_boolean(v.type):
                ns = s.isnull()
                if pd.api.types.is_string_dtype(s.dtype):
                    try:
                        s = s.str.lower() == "true"
                    except AttributeError:
                        s = s.fillna(0).astype(bool)
                else:
                    s = s.fillna(0).astype(bool)
                s = s.mask(ns, None)
            elif pa.types.is_integer(v.type):
                ns = s.isnull()
                s = (
                    s.fillna(0)
                    .astype(int)
                    .astype(to_single_pandas_dtype(v.type))
                    .mask(ns, None)
                )
            elif not pa.types.is_struct(v.type) and not pa.types.is_list(v.type):
                s = s.astype(to_single_pandas_dtype(v.type))
            data[v.name] = s
        return pd.DataFrame(data)

    def safe_groupby_apply(
        self,
        df: T,
        cols: List[str],
        func: Callable[[T], T],
        key_col_name="__safe_groupby_key__",
        **kwargs: Any,
    ) -> T:
        """Safe groupby apply operation on pandas like dataframes.
        In pandas like groupby apply, if any key is null, the whole group is dropped.
        This method makes sure those groups are included.

        :param df: pandas like dataframe
        :param cols: columns to group on, can be empty
        :param func: apply function, df in, df out
        :param key_col_name: temp key as index for groupu.
            default "__safe_groupby_key__"
        :return: output dataframe

        :Notice:
        The dataframe must be either empty, or with type pd.RangeIndex, pd.Int64Index
        or pd.UInt64Index and without a name, otherwise, `ValueError` will raise.
        """

        def _wrapper(keys: List[str], df: T) -> T:
            return func(df.drop(keys, axis=1).reset_index(drop=True))

        self.ensure_compatible(df)
        if len(cols) == 0:
            return func(df)
        params: Dict[str, Any] = {}
        for c in cols:
            params[key_col_name + "null_" + c] = df[c].isnull()
            params[key_col_name + "fill_" + c] = self.fillna_default(df[c])
        keys = list(params.keys())
        gdf = df.assign(**params)
        return (
            gdf.groupby(keys)
            .apply(lambda df: _wrapper(keys, df), **kwargs)
            .reset_index(drop=True)
        )

    def fillna_default(self, col: Any) -> Any:
        """Fill column with default values according to the dtype of the column.

        :param col: series of a pandas like dataframe
        :return: filled series
        """
        dtype = col.dtype
        if np.issubdtype(dtype, np.datetime64):
            return col.fillna(_DEFAULT_DATETIME)
        if np.issubdtype(dtype, np.str_) or np.issubdtype(
            dtype, np.string_
        ):  # pragma: no cover
            return col.fillna("")
        if np.issubdtype(dtype, np.bool_):
            return col.fillna(False)
        return col.fillna(0)

    def is_compatile_index(self, df: T) -> bool:
        """Check whether the datafame is compatible with the operations inside
        this utils collection

        :param df: pandas like dataframe
        :return: if it is compatible
        """
        return (
            isinstance(df.index, pd.RangeIndex) or df.index.inferred_type == "integer"
        )

    def ensure_compatible(self, df: T) -> None:
        """Check whether the datafame is compatible with the operations inside
        this utils collection, if not, it will raise ValueError

        :param df: pandas like dataframe
        :raises ValueError: if not compatible
        """
        if df.index.name is not None:
            raise ValueError("pandas like datafame index can't have name")
        if self.is_compatile_index(df):
            return
        if self.empty(df):
            return
        raise ValueError(
            f"pandas like datafame must have default index, but got {type(df.index)}"
        )


class PandasUtils(PandasLikeUtils[pd.DataFrame]):
    """A collection of pandas utils"""

    pass


PD_UTILS = PandasUtils()
