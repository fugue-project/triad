import json
from abc import ABC, abstractmethod
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Union, Callable

import pandas as pd
from triad.collections.dict import ParamDict
from triad.collections.schema import Schema


class DataFrame(ABC):
    SHOW_LOCK = RLock()

    def __init__(self, schema: Any = None, metadata: Any = None):
        if not callable(schema):
            schema = Schema(schema)
            schema.set_readonly()
            self._schema: Union[Schema, Callable[[], Schema]] = schema
            self._schema_discovered = True
        else:
            self._schema: Union[Schema, Callable[[], Schema]] = schema  # type: ignore
            self._schema_discovered = False
        self._metadata = ParamDict(metadata, deep=True)
        self._lazy_schema_lock = RLock()

    @property
    def metadata(self) -> ParamDict:
        return self._metadata

    @property
    def schema(self) -> Schema:
        if self._schema_discovered:
            # we must keep it simple because it could be called on every row by a user
            assert isinstance(self._schema, Schema)
            return self._schema  # type: ignore
        with self._lazy_schema_lock:
            self._schema = Schema(self._schema())  # type: ignore
            self._schema.set_readonly()
            self._schema_discovered = True
            return self._schema

    @abstractmethod
    def is_local(self) -> bool:  # pragma: no cover
        raise NotImplementedError

    # @abstractmethod
    # def as_local(self) -> "DataFrame":  # pragma: no cover
    #    raise NotImplementedError

    # @abstractmethod
    # def apply_schema(self, schema: Any) -> None:  # pragma: no cover
    #    raise NotImplementedError

    # @abstractmethod
    # def num_partitions(self) -> int:  # pragma: no cover
    #    raise NotImplementedError

    @abstractmethod
    def empty(self) -> bool:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def peek_array(self) -> Any:  # pragma: no cover
        raise NotImplementedError

    def peek_dict(self) -> Dict[str, Any]:
        arr = self.peek_array()
        return {self.schema.names[i]: arr[i] for i in range(len(self.schema))}

    @abstractmethod
    def count(self, persist: bool = False) -> int:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def as_pandas(self) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    # @abstractmethod
    # def as_pyarrow(self) -> pa.Table:  # pragma: no cover
    #    raise NotImplementedError

    @abstractmethod
    def as_array(
        self, columns: Optional[List[str]] = None, type_safe: bool = False
    ) -> List[Any]:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def as_array_iterable(
        self, columns: Optional[List[str]] = None, type_safe: bool = False
    ) -> Iterable[Any]:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def drop(self, cols: List[str]) -> "DataFrame":  # pragma: no cover
        raise NotImplementedError

    def show(
        self, n: int = 10, show_count: bool = False, title: Optional[str] = None
    ) -> None:
        arr: List[List[str]] = [str(self.schema).split(",")]
        count = -1
        m = 0
        for x in self.head(n):
            arr.append([str(t) for t in x])
            m += 1
        if m < n:
            count = m
        elif show_count:
            count = self.count()
        with DataFrame.SHOW_LOCK:
            if title is not None:
                print(title)
            print(type(self).__name__)
            # tb = PrettyTable(arr, 20)
            # print(tb)
            if count >= 0:
                print(f"Total count: {count}")
                print("")
            if len(self.metadata) > 0:
                print("Metadata:")
                print(self.metadata.to_json_str(indent=True))
                print("")

    def head(self, n: int, columns: Optional[List[str]] = None) -> List[Any]:
        res: List[Any] = []
        for row in self.as_array_iterable(columns, type_safe=True):
            if n < 1:
                break
            res.append(list(row))
            n -= 1
        return res

    def as_dict_iterable(
        self, columns: Optional[List[str]] = None
    ) -> Iterable[Dict[str, Any]]:
        if columns is None:
            columns = self.schema.names
        idx = range(len(columns))
        for x in self.as_array_iterable(columns, type_safe=True):
            yield {columns[i]: x[i] for i in idx}

    def get_info_str(self) -> str:
        return json.dumps(
            {
                "schema": str(self.schema),
                "type": "{}.{}".format(
                    self.__class__.__module__, self.__class__.__name__
                ),
                "metadata": self.metadata,
            }
        )


class LocalDataFrame(DataFrame):
    def __init__(self, schema: Any = None, metadata: Any = None):
        super().__init__(schema=schema, metadata=metadata)

    def is_local(self):
        return True
