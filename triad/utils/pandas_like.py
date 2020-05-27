from typing import Any, Callable, Iterable, List, Optional, TypeVar, Generic

import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.pyarrow import apply_schema, to_pandas_dtype

T = TypeVar("T", bound=Any)


class PandasLikeUtils(Generic[T]):
    def empty(self, df: T) -> bool:
        """Check if the dataframe is empty

        :param df: pandas like dataframe
        :return: if it is empty
        """
        return len(df.index) == 0

    def as_array_iterable(
        self,
        df: T,
        schema: Optional[pa.Schema] = None,
        type_safe: bool = False,
        null_safe: bool = False,
    ) -> Iterable[List[Any]]:
        """Convert pandas like dataframe to iterable of rows in the format of list.

        :param df: pandas like dataframe
        :param schema: columns and types for output.
            Leave it None to return all columns in original type
        :param type_safe: whether to enforce the types in schema, if not, it will
            return the original values from the dataframe
        :param null_safe: whether to ensure returning null for nan or null values in
            columns with type int, bool and string
        :return: iterable of rows, each row is a list

        :Notice:
        * `null_safe` by default is False, for non pandas dataframe, setting it to
        True may cause errors
        * If there are nested types in schema, the conversion can be a lot slower
        """
        if self.empty(df):
            return
        if schema is None:
            schema = self.to_schema(df)
        else:
            df = df[schema.names]
            orig = self.to_schema(df)
            if not orig.equals(schema):
                df = self.enforce_type(df, schema, null_safe)
        if not type_safe or all(not pa.types.is_nested(x) for x in schema.types):
            for arr in df.itertuples(index=False, name=None):
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
        if df.columns.dtype != "object":
            raise ValueError("Pandas dataframe must have named schema")
        if isinstance(df, pd.DataFrame) and len(df.index) > 0:
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

    def enforce_type(self, df: T, schema: pa.Schema, null_safe: bool = False) -> T:
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
        """
        if self.empty(df):
            return df
        if not null_safe:
            return df.astype(dtype=to_pandas_dtype(schema))
        for v in schema:
            s = df[v.name]
            if pa.types.is_string(v.type):
                ns = s[s.isnull()].index.tolist()
                s = s.astype(str)
                s.iloc[ns] = None
            elif pa.types.is_integer(v.type) or pa.types.is_boolean(v.type):
                ns = s[s.isnull()].index.tolist()
                s = s.fillna(0).astype(v.type.to_pandas_dtype())
                s.iloc[ns] = None
            elif not pa.types.is_struct(v.type):
                s = s.astype(v.type.to_pandas_dtype())
            df[v.name] = s
        return df

    def safe_groupby_apply(
        self,
        df: Any,
        cols: List[str],
        func: Callable[[T], T],
        key_col_name="__safe_groupby_key__",
        **kwargs: Any,
    ) -> T:
        """Safe groupby apply operation on pandas like dataframes

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
        self.ensure_compatible(df)
        if len(cols) == 0:
            return func(df)
        keys = df[cols].drop_duplicates().reset_index(drop=True)
        keys[key_col_name] = keys.index
        df = df.merge(keys, on=cols).set_index([key_col_name])

        def _wrapper(df: T) -> T:
            return func(df.reset_index(drop=True))

        return (
            df.groupby([key_col_name]).apply(_wrapper, **kwargs).reset_index(drop=True)
        )

    def is_compatile_index(self, df: T) -> bool:
        """Check whether the datafame is compatible with the operations inside
        this utils collection

        :param df: pandas like dataframe
        :return: if it is compatible
        """
        return isinstance(df.index, (pd.RangeIndex, pd.Int64Index, pd.UInt64Index))

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
    pass


PD_UTILS = PandasUtils()
