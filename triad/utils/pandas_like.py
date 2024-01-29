from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame

from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
    TRIAD_DEFAULT_TIMESTAMP_UNIT,
    cast_pa_table,
    pa_table_to_pandas,
    to_pa_datatype,
    to_pandas_dtype,
)

T = TypeVar("T", bound=Any)
ColT = TypeVar("ColT", bound=Any)

_DEFAULT_DATETIME = datetime(2000, 1, 1)
_ANTI_INDICATOR = "__anti_indicator__"
_CROSS_INDICATOR = "__corss_indicator__"


class PandasLikeUtils(Generic[T, ColT]):
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
        else:
            p = self.as_arrow(df, schema)
            d = p.to_pydict()
            cols = [d[n] for n in schema.names]
            for arr in zip(*cols):
                yield list(arr)

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
                    tp = df.dtypes.iloc[i]
                    if tp == np.dtype("object") or pd.api.types.is_string_dtype(tp):
                        t = pa.string()
                    elif isinstance(tp, pd.DatetimeTZDtype):
                        t = pa.timestamp(tp.unit, str(tp.tz))
                    else:
                        t = to_pa_datatype(tp)
                    yield pa.field(df.columns[i], t)

        fields: List[pa.Field] = []
        for field in get_fields():
            if pa.types.is_timestamp(field.type):
                fields.append(
                    pa.field(
                        field.name,
                        pa.timestamp(TRIAD_DEFAULT_TIMESTAMP_UNIT, field.type.tz),
                    )
                )
            elif pa.types.is_large_string(field.type):
                fields.append(pa.field(field.name, pa.string()))
            else:
                fields.append(field)
        return pa.schema(fields)

    def cast_df(
        self,
        df: T,
        schema: pa.Schema,
        use_extension_types: bool = True,
        use_arrow_dtype: bool = False,
        **kwargs: Any,
    ) -> T:
        """Cast pandas like dataframe to comply with ``schema``.

        :param df: pandas like dataframe
        :param schema: pyarrow schema to cast to
        :param use_extension_types: whether to use ``ExtensionDType``, default True
        :param use_arrow_dtype: whether to use ``ArrowDtype``, default False
        :param kwargs: other arguments passed to ``pa.Table.from_pandas``

        :return: converted dataframe
        """
        dtypes = to_pandas_dtype(
            schema,
            use_extension_types=use_extension_types,
            use_arrow_dtype=use_arrow_dtype,
        )
        if len(df) == 0:
            return pd.DataFrame({k: pd.Series(dtype=v) for k, v in dtypes.items()})
        if dtypes == df.dtypes.to_dict():
            return df
        adf = pa.Table.from_pandas(
            df, preserve_index=False, safe=False, **{"nthreads": 1, **kwargs}
        ).replace_schema_metadata()
        adf = cast_pa_table(adf, schema)
        return pa_table_to_pandas(
            adf,
            use_extension_types=use_extension_types,
            use_arrow_dtype=use_arrow_dtype,
        )

    def to_parquet_friendly(
        self, df: T, partition_cols: Optional[List[str]] = None
    ) -> T:
        """Parquet doesn't like pd.ArrowDtype(<nested types>), this function
        converts all nested types to object types

        :param df: the input dataframe
        :param partition_cols: the partition columns, if any, default None
        :return: the converted dataframe
        """
        pcols = partition_cols or []
        changed = False
        new_types: Dict[str, Any] = {}
        for k, v in df.dtypes.items():
            if k in pcols:
                new_types[k] = np.dtype(object)
                changed = True
            elif (
                hasattr(pd, "ArrowDtype")
                and isinstance(v, pd.ArrowDtype)
                and pa.types.is_nested(v.pyarrow_dtype)
            ):
                new_types[k] = np.dtype(object)
                changed = True
            else:
                new_types[k] = v
        if changed:
            df = df.astype(new_types)
        return df

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

        def _wrapper(df: T) -> T:
            return func(df.reset_index(drop=True))

        self.ensure_compatible(df)
        if len(cols) == 0:
            return func(df)
        return (
            df.groupby(cols, dropna=False, group_keys=False)
            .apply(lambda df: _wrapper(df), **kwargs)
            .reset_index(drop=True)
        )

    def fillna_default(self, col: Any) -> Any:
        """Fill column with default values according to the dtype of the column.

        :param col: series of a pandas like dataframe
        :return: filled series
        """
        dtype = col.dtype
        if pd.api.types.is_datetime64_dtype(dtype):
            return col.fillna(_DEFAULT_DATETIME)
        if pd.api.types.is_string_dtype(dtype):
            return col.fillna("")
        if pd.api.types.is_bool_dtype(dtype):
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
        if self.empty(df):  # for pandas < 2
            return  # pragma: no cover
        raise ValueError(
            f"pandas like datafame must have default index, but got {type(df.index)}"
        )

    def parse_join_type(self, join_type: str) -> str:
        """Parse join type string to standard join type string

        :param join_type: the join type string
        :return: the standard join type string
        """
        join_type = join_type.replace(" ", "").replace("_", "").lower()
        if join_type in ["inner", "cross"]:
            return join_type
        if join_type in ["inner", "join"]:
            return "inner"
        if join_type in ["semi", "leftsemi"]:
            return "left_semi"
        if join_type in ["anti", "leftanti"]:
            return "left_anti"
        if join_type in ["left", "leftouter"]:
            return "left_outer"
        if join_type in ["right", "rightouter"]:
            return "right_outer"
        if join_type in ["outer", "full", "fullouter"]:
            return "full_outer"
        raise NotImplementedError(join_type)

    def drop_duplicates(self, df: T) -> T:
        """Remove duplicated rows

        :param df: the dataframe
        :return: the dataframe without duplicated rows
        """
        return df.drop_duplicates()

    def union(self, ndf1: T, ndf2: T, unique: bool) -> T:
        """Union two dataframes

        :param ndf1: dataframe 1
        :param ndf2: dataframe 2
        :param unique: whether to remove duplicated rows
        :return: the unioned dataframe
        """
        ndf1, ndf2 = self._preprocess_set_op(ndf1, ndf2)
        ndf = self.concat_dfs(ndf1, ndf2)
        if unique:
            ndf = self.drop_duplicates(ndf)
        return ndf

    def intersect(self, df1: T, df2: T, unique: bool) -> T:
        """Intersect two dataframes

        :param ndf1: dataframe 1
        :param ndf2: dataframe 2
        :param unique: whether to remove duplicated rows
        :return: the intersected dataframe
        """
        ndf1, ndf2 = self._preprocess_set_op(df1, df2)
        ndf = ndf1.merge(self.drop_duplicates(ndf2))
        if unique:
            ndf = self.drop_duplicates(ndf)
        return ndf

    def except_df(
        self,
        df1: T,
        df2: T,
        unique: bool,
        anti_indicator_col: str = _ANTI_INDICATOR,
    ) -> T:
        """Remove df2 from df1

        :param df1: dataframe 1
        :param df2: dataframe 2
        :param unique: whether to remove duplicated rows in the result
        :return: the dataframe with df2 removed
        """
        ndf1, ndf2 = self._preprocess_set_op(df1, df2)
        # TODO: lack of test to make sure original ndf2 is not changed
        ndf2 = self._with_indicator(ndf2, anti_indicator_col)
        ndf = ndf1.merge(ndf2, how="left", on=list(ndf1.columns))
        ndf = ndf[ndf[anti_indicator_col].isnull()].drop([anti_indicator_col], axis=1)
        if unique:
            ndf = self.drop_duplicates(ndf)
        return ndf

    def join(
        self,
        ndf1: T,
        ndf2: T,
        join_type: str,
        on: List[str],
        anti_indicator_col: str = _ANTI_INDICATOR,
        cross_indicator_col: str = _CROSS_INDICATOR,
    ) -> T:
        """Join two dataframes

        :param ndf1: dataframe 1
        :param ndf2: dataframe 2
        :param join_type: join type, can be inner, left_semi, left_anti, left_outer,
            right_outer, full_outer, cross
        :param on: join keys

        :return: the joined dataframe
        """
        join_type = self.parse_join_type(join_type)
        if join_type == "inner":
            ndf1 = ndf1.dropna(subset=on)
            ndf2 = ndf2.dropna(subset=on)
            joined = ndf1.merge(ndf2, how=join_type, on=on)
        elif join_type == "left_semi":
            ndf1 = ndf1.dropna(subset=on)
            ndf2 = self.drop_duplicates(ndf2[on].dropna())
            joined = ndf1.merge(ndf2, how="inner", on=on)
        elif join_type == "left_anti":
            # TODO: lack of test to make sure original ndf2 is not changed
            ndf2 = self.drop_duplicates(ndf2[on].dropna())
            ndf2 = self._with_indicator(ndf2, anti_indicator_col)
            joined = ndf1.merge(ndf2, how="left", on=on)
            joined = joined[joined[anti_indicator_col].isnull()].drop(
                [anti_indicator_col], axis=1
            )
        elif join_type == "left_outer":
            ndf2 = ndf2.dropna(subset=on)
            joined = ndf1.merge(ndf2, how="left", on=on)
        elif join_type == "right_outer":
            ndf1 = ndf1.dropna(subset=on)
            joined = ndf1.merge(ndf2, how="right", on=on)
        elif join_type == "full_outer":
            add: List[str] = []
            for f in on:
                name = f + "_null"
                s1 = ndf1[f].isnull().astype(int)
                ndf1[name] = s1
                s2 = ndf2[f].isnull().astype(int) * 2
                ndf2[name] = s2
                add.append(name)
            joined = ndf1.merge(ndf2, how="outer", on=on + add).drop(add, axis=1)
        elif join_type == "cross":
            assert_or_throw(
                len(on) == 0, ValueError(f"cross join can't have join keys {on}")
            )
            ndf1 = self._with_indicator(ndf1, cross_indicator_col)
            ndf2 = self._with_indicator(ndf2, cross_indicator_col)
            joined = ndf1.merge(ndf2, how="inner", on=[cross_indicator_col]).drop(
                [cross_indicator_col], axis=1
            )
        else:  # pragma: no cover
            raise NotImplementedError(join_type)
        return joined

    def concat_dfs(self, *dfs: T) -> T:  # pragma: no cover
        """Concatenate dataframes

        :param dfs: the dataframes to concatenate
        :return: the concatenated dataframe
        """
        raise NotImplementedError

    def _preprocess_set_op(self, ndf1: T, ndf2: T) -> Tuple[T, T]:
        assert_or_throw(
            len(list(ndf1.columns)) == len(list(ndf2.columns)),
            ValueError("two dataframes have different number of columns"),
        )
        ndf2.columns = ndf1.columns  # this is SQL behavior
        return ndf1, ndf2

    def _with_indicator(self, df: T, name: str) -> T:
        return df.assign(**{name: 1})


class PandasUtils(PandasLikeUtils[pd.DataFrame, pd.Series]):
    """A collection of pandas utils"""

    def concat_dfs(self, *dfs: DataFrame) -> DataFrame:
        return pd.concat(list(dfs))


PD_UTILS = PandasUtils()
