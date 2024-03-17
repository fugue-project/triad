from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

from triad.collections.dict import IndexedOrderedDict
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import (
    expression_to_schema,
    is_supported,
    schema_to_expression,
    pa_schemas_equal,
    to_pa_datatype,
    to_pandas_dtype,
)
from triad.utils.schema import (
    quote_name,
    safe_replace_out_of_quote,
    safe_search_out_of_quote,
    safe_split_and_unquote,
)


class SchemaError(Exception):
    """Exceptions related with construction and modifying schemas"""

    def __init__(self, message: Any):
        super().__init__(message)


class Schema(IndexedOrderedDict[str, pa.Field]):
    """A Schema wrapper on top of pyarrow.Fields. This has more features
    than pyarrow.Schema, and they can convert to each other.

    This class can be initialized from schema like objects. Here is a list of
    schema like objects:

    * pyarrow.Schema or Schema objects
    * pyarrow.Field: single field will be treated as a single column schema
    * schema expressions: :func:`~triad.utils.pyarrow.expression_to_schema`
    * Dict[str,Any]: key will be the columns, and value will be type like objects
    * Tuple[str,Any]: first item will be the only column name of the schema,
      and the second has to be a type like object
    * List[Any]: a list of Schema like objects
    * pandas.DataFrame: it will extract the dataframe's schema

    Here is a list of data type like objects:

    * pyarrow.DataType
    * pyarrow.Field: will only use the type attribute of the field
    * type expression or other objects: for :func:`~triad.utils.pyarrow.to_pa_datatype`

    .. admonition:: Examples

        .. code-block:: python

            Schema("a:int,b:int")
            Schema("a:int","b:int")
            Schema(a=int,b=str) # == Schema("a:long,b:str")
            Schema(dict(a=int,b=str)) # == Schema("a:long,b:str")
            Schema([(a,int),(b,str)]) # == Schema("a:long,b:str")
            Schema((a,int),(b,str)) # == Schema("a:long,b:str")
            Schema("a:[int],b:{x:int,y:{z:[str],w:byte}},c:[{x:str}]")

    .. note::

        * For supported pyarrow.DataTypes see :func:`~triad.utils.pyarrow.is_supported`
        * If you use python type as data type (e.g. `Schema(a=int,b=str)`) be aware
          the data type different. (e.g. python `int` type -> pyarrow `long`/`int64`
          type)
        * When not readonly, only `append` is allowed, `update` or `remove` are
          disallowed
        * When readonly, no modification on the existing schema is allowed
        * `append`, `update` and `remove` are always allowed when creating a new object
        * InvalidOperationError will be raised for disallowed operations
        * At most one of `*args` and `**kwargs` can be set

    :param args: one or multiple schema like objects, which will be combined in order
    :param kwargs: key value pairs for the schema
    """

    def __init__(self, *args: Any, **kwargs: Any):
        if len(args) > 0 and len(kwargs) > 0:
            raise SchemaError("Can't set both *args and **kwargs")
        if len(args) == 1:  # duplicate code for better performance
            if isinstance(args[0], Schema):
                super().__init__(args[0])  # type: ignore
                return
            fields: Optional[List[pa.Field]] = None
            if isinstance(args[0], str):
                fields = list(expression_to_schema(args[0]))
            if isinstance(args[0], pa.Schema):
                fields = list(args[0])
            if isinstance(args[0], pa.Field):
                fields = [args[0]]
            if fields is not None:
                fields = [self._validate_field(f) for f in fields]
                super().__init__([(x.name, x) for x in fields])
                return
        super().__init__()
        if len(args) > 0:
            self.append(list(args))
        elif len(kwargs) > 0:
            self.append(kwargs)

    @property
    def names(self) -> List[str]:
        """List of column names"""
        self._build_index()
        return self._index_key  # type: ignore

    @property
    def fields(self) -> List[pa.Field]:
        """List of pyarrow.Fields"""
        return list(self.values())

    @property
    def types(self) -> List[pa.DataType]:
        """List of pyarrow.DataTypes"""
        return [v.type for v in self.values()]

    @property
    def pyarrow_schema(self) -> pa.Schema:
        """convert as pyarrow.Schema"""
        return pa.schema(self.fields)

    @property
    def pa_schema(self) -> pa.Schema:
        """convert as pyarrow.Schema"""
        return self.pyarrow_schema

    @property
    def pandas_dtype(self) -> Dict[str, np.dtype]:
        """Convert as `dtype` dict for pandas dataframes.
        Currently, struct type is not supported
        """
        return self.to_pandas_dtype(self.pa_schema)

    def to_pandas_dtype(
        self, use_extension_types: bool = False, use_arrow_dtype: bool = False
    ) -> Dict[str, np.dtype]:
        """Convert as `dtype` dict for pandas dataframes.

        :param use_extension_types: if True, use pandas extension types,
            default False
        :param use_arrow_dtype: if True and when pandas supports ``ArrowDType``,
            use pyarrow types, default False

        .. note::

            * If ``use_extension_types`` is False and ``use_arrow_dtype`` is True,
                it converts all types to ``ArrowDType``
            * If both are true, it converts types to the numpy backend nullable
                dtypes if possible, otherwise, it converts to ``ArrowDType``
        """
        return to_pandas_dtype(
            self.pa_schema,
            use_extension_types=use_extension_types,
            use_arrow_dtype=use_arrow_dtype,
        )

    @property
    def pd_dtype(self) -> Dict[str, np.dtype]:
        """convert as `dtype` dict for pandas dataframes.
        Currently, struct type is not supported
        """
        return self.pandas_dtype

    def assert_not_empty(self) -> "Schema":
        """Raise exception if schema is empty"""
        if len(self) > 0:
            return self
        raise SchemaError("Schema can't be empty")

    def copy(self) -> "Schema":
        """Clone Schema object

        :return: cloned object
        """
        other = super().copy()
        assert isinstance(other, Schema)
        return other

    def __repr__(self) -> str:
        """Convert to the string representation of the schema"""
        return schema_to_expression(self.pyarrow_schema)

    def __iadd__(self, obj: Any) -> "Schema":
        """Append a schema like object to the current schema

        :param obj: a schema like object

        .. admonition:: Examples

            .. code-block:: python

                s = Schema("a:int,b:str")
                s += "c:long"
                s += ("d",int)
        """
        return self.append(obj)

    def __isub__(self, obj: Any) -> "Schema":
        """Remove columns from a schema is not allowed"""
        raise SchemaError("'-=' is not allowed for Schema")

    def __add__(self, obj: Any) -> "Schema":
        """Add a schema like object to the current schema

        :param obj: a schema like object

        .. admonition:: Examples

            .. code-block:: python

                s = Schema("a:int,b:str")
                s = s + "c:long" +  ("d",int)
        """
        return self.copy().append(obj)

    def __sub__(self, obj: Any) -> "Schema":
        """Remove columns or schema from the schema.

        :param obj: one column name, a list/set of column names or
            a schema like object
        :return: a schema excluding the columns in ``obj``

        .. note::

            If ``obj`` violates any of the following conditions, ``SchemaError``
            will be raised:

            * all columns in ``obj`` must be found in the schema
            * If ``obj`` is a schema like object, the types must also match
        """
        return self.remove(
            obj,
            ignore_key_mismatch=False,
            require_type_match=True,
            ignore_type_mismatch=False,
        )

    def __setitem__(  # type: ignore
        self, name: str, value: Any, *args: List[Any], **kwds: Dict[str, Any]
    ) -> None:
        assert_arg_not_none(value, "value")
        if name in self:  # update existing value is not allowed
            raise SchemaError(f"{name} already exists in {self}")
        if isinstance(value, pa.Field):
            assert_or_throw(
                name == value.name, SchemaError(f"{name} doesn't match {value}")
            )
        elif isinstance(value, pa.DataType):
            value = pa.field(name, value)
        else:
            value = pa.field(name, to_pa_datatype(value))
        assert_or_throw(
            is_supported(value.type), SchemaError(f"{value} is not supported")
        )
        super().__setitem__(name, value, *args, **kwds)  # type: ignore

    def __eq__(self, other: Any) -> bool:
        """Check if the two schemas are equal

        :param other: a schema like object
        :return: True if the two schemas are equal
        """
        return self.is_like(other)

    def __contains__(self, key: Any) -> bool:  # noqa: C901
        """Check if the schema contains the key.

        :param key: a column name, a list of column names, a data
            type like object or a schema like object
        :return: True if the schema contains the object
        """
        if key is None:
            return False
        if isinstance(key, str):
            if ":" not in key:
                return super().__contains__(key)
        elif isinstance(key, pa.Field):
            res = super().get(key.name, None)
            if res is None:
                return False
            return key.type == res.type
        elif isinstance(key, Schema):
            for f in key.values():
                if f not in self:
                    return False
            return True
        elif isinstance(key, List):
            for f in key:
                if f not in self:
                    return False
            return True
        return Schema(key) in self

    def is_like(
        self,
        other: Any,
        equal_groups: Optional[List[List[Callable[[pa.DataType], bool]]]] = None,
    ) -> bool:
        """Check if the two schemas are equal or similar

        :param other: a schema like object
        :param equal_groups: a list of list of functions to check if two types
            are equal, default None

        :return: True if the two schemas are equal

        .. admonition:: Examples

            .. code-block:: python

                s = Schema("a:int,b:str")
                assert s.is_like("a:int,b:str")
                assert not s.is_like("a:long,b:str")
                assert s.is_like("a:long,b:str", equal_groups=[(pa.types.is_integer,)])
        """
        if other is None:
            return False
        if other is self:
            return True
        if isinstance(other, Schema):
            _other = other
        elif isinstance(other, str):
            if equal_groups is None:
                return self.__repr__() == other
            _other = Schema(other)
        else:
            _other = Schema(other)
        return pa_schemas_equal(
            self.pa_schema, _other.pa_schema, equal_groups=equal_groups
        )

    def append(self, obj: Any) -> "Schema":  # noqa: C901
        """Append schema like object to the current schema. Only new columns
        are allowed.

        :raises SchemaError: if a column exists or is invalid or obj is not convertible
        :return: the Schema object itself
        """
        try:
            if obj is None:
                return self
            elif isinstance(obj, pa.Field):
                self[obj.name] = obj.type
            elif isinstance(obj, str):
                self._append_pa_schema(expression_to_schema(obj))
            elif isinstance(obj, Dict):
                for k, v in obj.items():
                    self[k] = v
            elif isinstance(obj, pa.Schema):
                self._append_pa_schema(obj)
            elif isinstance(obj, pd.DataFrame):
                self._append_pa_schema(PD_UTILS.to_schema(obj))
            elif isinstance(obj, Tuple):  # type: ignore
                self[obj[0]] = obj[1]
            elif isinstance(obj, List):
                for x in obj:
                    self.append(x)
            else:
                raise SchemaError(f"Invalid schema to add {obj}")
            return self
        except SchemaError:
            raise
        except Exception as e:
            raise SchemaError(str(e))

    def remove(  # noqa: C901
        self,
        obj: Any,
        ignore_key_mismatch: bool = False,
        require_type_match: bool = True,
        ignore_type_mismatch: bool = False,
    ) -> "Schema":
        """Remove columns or schema from the schema

        :param obj: one column name, a list/set of column names or
            a schema like object
        :param ignore_key_mismatch: if True, ignore the non-existing keys,
            default False
        :param require_type_match: if True, a match requires the same key
            and same type (if ``obj`` contains type), otherwise, only the
            key needs to match, default True
        :param ignore_type_mismatch: if False, when keys match but types don't
            (if ``obj`` contains type), raise an exception ``SchemaError``,
            default False
        :return: a schema excluding the columns in ``obj``
        """
        if obj is None:
            return self.copy()
        target = self
        if isinstance(obj, str):
            if ":" in obj:  # expression
                ps = expression_to_schema(obj)
                pairs: List[Tuple[str, pa.DataType]] = list(zip(ps.names, ps.types))
            else:
                pairs = [(obj, None)]  # single key
        elif isinstance(obj, (pa.Schema, Schema)):
            pairs = list(zip(obj.names, obj.types))
        elif isinstance(obj, (List, Set)):
            keys: List[str] = []
            other: List[Any] = []
            for x in obj:
                if isinstance(x, str) and ":" not in x:
                    keys.append(x)
                else:
                    other.append(x)
            pairs = [(x, None) for x in keys]
            for o in other:
                target = target.remove(
                    o,
                    ignore_key_mismatch=ignore_key_mismatch,
                    require_type_match=require_type_match,
                    ignore_type_mismatch=ignore_type_mismatch,
                )
        else:
            return self.remove(
                Schema(obj),
                ignore_key_mismatch=ignore_key_mismatch,
                require_type_match=require_type_match,
                ignore_type_mismatch=ignore_type_mismatch,
            )
        od = OrderedDict(target)
        for k, v in pairs:
            # k = k.strip()
            # if k == "":
            #    continue
            if k not in od:
                if ignore_key_mismatch:
                    continue
                raise SchemaError(
                    f"Can't remove {quote_name(k)} from {target} (not found)"
                )
            if v is None:
                del od[k]
            else:
                tp = od[k].type
                if not require_type_match or tp == v:
                    del od[k]
                elif not ignore_type_mismatch:
                    raise SchemaError(
                        f"Unable to remove {k}:{v} from {self}, type mismatch"
                    )
        return Schema(od)

    def alter(self, subschema: Any) -> "Schema":
        """Alter the schema with a subschema

        :param subschema: a schema like object
        :return: the altered schema
        """
        if subschema is None:
            return self
        sub = Schema(subschema)
        assert_or_throw(
            sub.names in self,
            lambda: ValueError(f"{sub.names} are not all in {self}"),
        )
        return Schema([(k, sub.get(k, v)) for k, v in self.items()])

    def extract(  # noqa: C901
        self,
        obj: Any,
        ignore_key_mismatch: bool = False,
        require_type_match: bool = True,
        ignore_type_mismatch: bool = False,
    ) -> "Schema":
        """Extract a sub schema from the schema based on the columns in ``obj``

        :param obj: one column name, a list/set of column names or
            a schema like object
        :param ignore_key_mismatch: if True, ignore the non-existing keys,
            default False
        :param require_type_match: if True, a match requires the same key
            and same type (if ``obj`` contains type), otherwise, only the
            key needs to match, default True
        :param ignore_type_mismatch: if False, when keys match but types don't
            (if ``obj`` contains type), raise an exception ``SchemaError``,
            default False
        :return: a sub-schema containing the columns in ``obj``
        """
        if obj is None:
            return Schema()
        if isinstance(obj, str):
            if ":" in obj:  # expression
                ps = expression_to_schema(obj)
                pairs: List[Tuple[str, pa.DataType]] = list(zip(ps.names, ps.types))
            else:
                pairs = [(obj, None)]  # single key
        elif isinstance(obj, (pa.Schema, Schema)):
            pairs = list(zip(obj.names, obj.types))
        elif isinstance(obj, List):
            fields: List[pa.Field] = []
            for x in obj:
                if isinstance(x, str) and ":" not in x:
                    if x not in self:
                        if not ignore_key_mismatch:
                            raise SchemaError(f"Can't extract {x} from {self}")
                    else:
                        fields.append(self[x])
                else:
                    fields += self.extract(
                        x,
                        ignore_key_mismatch=ignore_key_mismatch,
                        require_type_match=require_type_match,
                        ignore_type_mismatch=ignore_type_mismatch,
                    ).fields
            return Schema(pa.schema(fields))
        else:
            return self.extract(
                Schema(obj),
                ignore_key_mismatch=ignore_key_mismatch,
                require_type_match=require_type_match,
                ignore_type_mismatch=ignore_type_mismatch,
            )
        fields = []
        for k, v in pairs:
            if k not in self:
                if ignore_key_mismatch:
                    continue
                raise SchemaError(f"Can't extract {k} from {self}")
            if v is None:
                fields.append(self[k])
            else:
                tp = self[k].type
                if not require_type_match or tp == v:
                    fields.append(self[k])
                elif not ignore_type_mismatch:
                    raise SchemaError(
                        f"Unable to extract {k}:{v} from {self}, type mismatch"
                    )
        return Schema(pa.schema(fields))

    def exclude(
        self,
        other: Any,
        require_type_match: bool = True,
        ignore_type_mismatch: bool = False,
    ) -> "Schema":
        """Exclude columns from the current schema which are also in ``other``.
        ``other`` can contain columns that are not in the current schema, they
        will be ignored.

        :param other: one column name, a list/set of column names or
            a schema like object
        :param require_type_match: if True, a match requires the same key
            and same type (if ``obj`` contains type), otherwise, only the
            key needs to match, default True
        :param ignore_type_mismatch: if False, when keys match but types don't
            (if ``obj`` contains type), raise an exception ``SchemaError``,
            default False
        :return: a schema excluding the columns in ``other``
        """
        return self.remove(
            other,
            ignore_key_mismatch=True,
            require_type_match=require_type_match,
            ignore_type_mismatch=ignore_type_mismatch,
        )

    def intersect(
        self,
        other: Any,
        require_type_match: bool = True,
        ignore_type_mismatch: bool = True,
        use_other_order: bool = False,
    ) -> "Schema":
        """Extract the sub-schema from the current schema which are also in
        ``other``. ``other`` can contain columns that are not in the current schema,
        they will be ignored.

        :param other: one column name, a list/set of column names or
            a schema like object
        :param require_type_match: if True, a match requires the same key
            and same type (if ``obj`` contains type), otherwise, only the
            key needs to match, default True
        :param ignore_type_mismatch: if False, when keys match but types don't
            (if ``obj`` contains type), raise an exception ``SchemaError``,
            default False
        :param use_other_order: if True, the output schema will use the column order
            of ``other``, default False
        :return: the intersected schema
        """
        if not use_other_order:
            diff = self.exclude(
                other,
                require_type_match=require_type_match,
                ignore_type_mismatch=ignore_type_mismatch,
            )
            return self - diff
        else:
            return self.extract(
                other,
                require_type_match=require_type_match,
                ignore_type_mismatch=ignore_type_mismatch,
            )

    def union(self, other: Any, require_type_match: bool = False) -> "Schema":
        """Union the ``other`` schema

        :param other: a schema like object
        :param require_type_match: if True, a match requires the same key
            and same type (if ``obj`` contains type), otherwise, only the
            key needs to match, default True
        :return: the new unioned schema
        """
        return self.copy().union_with(other, require_type_match=require_type_match)

    def union_with(self, other: Any, require_type_match: bool = False) -> "Schema":
        """Union the ``other`` schema into the current schema

        :param other: a schema like object
        :param require_type_match: if True, a match requires the same key
            and same type (if ``obj`` contains type), otherwise, only the
            key needs to match, default True
        :return: the current schema
        """
        if not isinstance(other, Schema):
            other = Schema(other)
        try:
            other = other.exclude(
                self, require_type_match=require_type_match, ignore_type_mismatch=False
            )
            self += other
            return self
        except Exception as e:
            raise SchemaError(f"Unable to union {self} with {other}: {str(e)}")

    def rename(self, columns: Dict[str, str], ignore_missing: bool = False) -> "Schema":
        """Rename the current schema and generate a new one

        :param columns: dictionary to map from old to new column names
        :return: renamed schema object
        """
        if not ignore_missing:
            for x in columns:
                if x not in self:
                    raise SchemaError(f"Failed to rename: {x} not in {self}")
        pairs = [
            (k if k not in columns else columns[k], v.type) for k, v in self.items()
        ]
        return Schema(pairs)

    def transform(self, *args: Any, **kwargs: Any) -> "Schema":
        """Transform the current schema to a new schema

        :raises SchemaError: if there is any exception
        :return: transformed schema

        .. admonition:: Examples

            .. code-block:: python

                s=Schema("a:int,b:int,c:str")
                s.transform("x:str") # x:str
                # add
                s.transform("*,x:str") # a:int,b:int,c:str,x:str
                s.transform("*","x:str") # a:int,b:int,c:str,x:str
                s.transform("*",x=str) # a:int,b:int,c:str,x:str
                # subtract
                s.transform("*-c,a") # b:int
                s.transform("*-c-a") # b:int
                s.transform("*~c,a,x") # b:int  # ~ means exlcude if exists
                s.transform("*~c~a~x") # b:int  # ~ means exlcude if exists
                # + means overwrite existing and append new
                s.transform("*+e:str,b:str,d:str") # a:int,b:str,c:str,e:str,d:str
                # you can have multiple operations
                s.transform("*+b:str-a") # b:str,c:str
                # callable
                s.transform(lambda s:s.fields[0]) # a:int
                s.transform(lambda s:s.fields[0], lambda s:s.fields[2]) # a:int,c:str
        """
        try:
            result = Schema()
            for a in args:
                if callable(a):
                    result += a(self)
                elif isinstance(a, str):
                    op_pos = [x[0] for x in safe_search_out_of_quote(a, "-~+")]
                    op_pos.append(len(a))
                    s = Schema(
                        safe_replace_out_of_quote(a[: op_pos[0]], "*", str(self))
                    )
                    for i in range(0, len(op_pos) - 1):
                        op, expr = a[op_pos[i]], a[(op_pos[i] + 1) : op_pos[i + 1]]
                        if op in ["-", "~"]:
                            cols = safe_split_and_unquote(
                                expr, ",", on_unquoted_empty="ignore"
                            )
                            s = s.exclude(cols) if op == "~" else s - cols
                        else:  # +
                            overwrite = Schema(expr)
                            s = Schema(
                                [(k, overwrite.get(k, v)) for k, v in s.items()]
                            ).union_with(overwrite)
                    result += s
                else:
                    result += a
            return result + Schema(kwargs)
        except SchemaError:
            raise
        except Exception as e:
            raise SchemaError(e)

    def create_empty_arrow_table(self) -> pa.Table:
        """Create an empty pyarrow table based on the schema"""
        if not hasattr(pa.Table, "from_pylist"):  # pragma: no cover
            arr = [pa.array([])] * len(self)
            return pa.Table.from_arrays(arr, schema=self.pa_schema)
        return pa.Table.from_pylist([], schema=self.pa_schema)

    def create_empty_pandas_df(
        self, use_extension_types: bool = False, use_arrow_dtype: bool = False
    ) -> pd.DataFrame:
        """Create an empty pandas dataframe based on the schema

        :param use_extension_types: if True, use pandas extension types,
            default False
        :param use_arrow_dtype: if True and when pandas supports ``ArrowDType``,
            use pyarrow types, default False
        :return: empty pandas dataframe
        """
        dtypes = self.to_pandas_dtype(
            use_extension_types=use_extension_types,
            use_arrow_dtype=use_arrow_dtype,
        )
        return pd.DataFrame({k: pd.Series(dtype=v) for k, v in dtypes.items()})

    def _pre_update(self, op: str, need_reindex: bool = True) -> None:
        if op == "__setitem__":
            super()._pre_update(op, need_reindex)
        else:
            raise SchemaError(f"{op} is not allowed in Schema")

    def _append_pa_schema(self, other: pa.Schema) -> "Schema":
        for k, v in zip(other.names, other.types):
            self[k] = v
        return self

    def _validate_field(self, field: pa.Field) -> pa.Field:
        assert_or_throw(
            is_supported(field.type), SchemaError(f"{field} type is not supported")
        )
        return field
