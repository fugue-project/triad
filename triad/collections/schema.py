from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
from triad.collections.dict import IndexedOrderedDict
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pyarrow import (
    expression_to_schema,
    is_supported,
    schema_to_expression,
    to_pa_datatype,
    to_pandas_dtype,
    validate_column_name,
)
from triad.utils.pandas_like import PD_UTILS


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

    The column name must follow the rules
    by :func:`~triad.utils.pyarrow.validate_column_name`

    :Examples:
    >>> Schema("a:int,b:int")
    >>> Schema("a:int","b:int")
    >>> Schema(a=int,b=str) # == Schema("a:long,b:str")
    >>> Schema(dict(a=int,b=str)) # == Schema("a:long,b:str")
    >>> Schema([(a,int),(b,str)]) # == Schema("a:long,b:str")
    >>> Schema((a,int),(b,str)) # == Schema("a:long,b:str")
    >>> Schema("a:[int],b:{x:int,y:{z:[str],w:byte}},c:[{x:str}]")

    :Notice:
    * For supported pyarrow.DataTypes see :func:`~triad.utils.pyarrow.is_supported`
    * If you use python type as data type (e.g. `Schema(a=int,b=str)`) be aware
        the data type different. (e.g. python `int` type -> pyarrow `long`/`int64` type)
    * When not readonly, only `append` is allowed, `update` or `remove` are disallowed
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
        """convert as `dtype` dict for pandas dataframes.
        Currently, struct type is not supported
        """
        return to_pandas_dtype(self.pa_schema)

    @property
    def pd_dtype(self) -> Dict[str, np.dtype]:
        """convert as `dtype` dict for pandas dataframes.
        Currently, struct type is not supported
        """
        return self.pandas_dtype

    def assert_not_empty(self) -> "Schema":
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
        return schema_to_expression(self.pyarrow_schema)

    def __iadd__(self, obj: Any) -> "Schema":
        return self.append(obj)

    def __isub__(self, obj: Any) -> "Schema":
        raise SchemaError("'-=' is not allowed for Schema")

    def __add__(self, obj: Any) -> "Schema":
        return self.copy().append(obj)

    def __sub__(self, obj: Any) -> "Schema":
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
        if not validate_column_name(name):
            raise SchemaError(f"Invalid column name {name}")
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
        if other is None:
            return False
        if isinstance(other, Schema):
            return super().__eq__(other)
        if isinstance(other, str):
            return self.__repr__() == other
        return self == Schema(other)

    def __contains__(self, key: Any) -> bool:  # noqa: C901
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
            k = k.strip()
            if k == "":
                continue
            if k not in od:
                if ignore_key_mismatch:
                    continue
                raise SchemaError(f"Can't remove {k} from {target}")
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

    def extract(  # noqa: C901
        self,
        obj: Any,
        ignore_key_mismatch: bool = False,
        require_type_match: bool = True,
        ignore_type_mismatch: bool = False,
    ) -> "Schema":
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
            k = k.strip()
            if k == "":
                continue
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
        return self.copy().union_with(other, require_type_match=require_type_match)

    def union_with(self, other: Any, require_type_match: bool = False) -> "Schema":
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

        :Examples:
        >>> s=Schema("a:int,b:int,c:str")
        >>> s.transform("x:str") # x:str
        >>> # add
        >>> s.transform("*,x:str") # a:int,b:int,c:str,x:str
        >>> s.transform("*","x:str") # a:int,b:int,c:str,x:str
        >>> s.transform("*",x=str) # a:int,b:int,c:str,x:str
        >>> # subtract
        >>> s.transform("*-c,a") # b:int
        >>> s.transform("*-c-a") # b:int
        >>> s.transform("*~c,a,x") # b:int  # ~ means exlcude if exists
        >>> s.transform("*~c~a~x") # b:int  # ~ means exlcude if exists
        >>> # + means overwrite existing and append new
        >>> s.transform("*+e:str,b:str,d:str") # a:int,b:str,c:str,e:str,d:str
        >>> # you can have multiple operations
        >>> s.transform("*+b:str-a") # b:str,c:str
        >>> # callable
        >>> s.transform(lambda s:s.fields[0]) # a:int
        >>> s.transform(lambda s:s.fields[0], lambda s:s.fields[2]) # a:int,c:str
        """
        try:
            result = Schema()
            for a in args:
                if callable(a):
                    result += a(self)
                elif isinstance(a, str):
                    op_pos = [i for i, c in enumerate(a) if c in ["-", "~", "+"]]
                    op_pos.append(len(a))
                    s = Schema(a[: op_pos[0]].replace("*", str(self)))
                    for i in range(0, len(op_pos) - 1):
                        op, expr = a[op_pos[i]], a[(op_pos[i] + 1) : op_pos[i + 1]]
                        if op in ["-", "~"]:
                            cols = [
                                x.strip() for x in expr.split(",") if x.strip() != ""
                            ]
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
            validate_column_name(field.name), SchemaError(f"{field} name is invalid")
        )
        assert_or_throw(
            is_supported(field.type), SchemaError(f"{field} type is not supported")
        )
        return field
