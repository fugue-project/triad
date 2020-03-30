import copy
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from triad.utils.convert import as_type
from triad.utils.iter import to_kv_iterable


class IndexedOrderedDict(OrderedDict):
    def __init__(self, *args: List[Any], **kwds: Dict[str, Any]):
        super().__init__(*args, **kwds)
        self._need_reindex = True
        self._key_index: Dict[Any, int] = {}
        self._index_key: List[Any] = []

    def index_of_key(self, key: Any) -> int:
        self._build_index()
        return self._key_index[key]

    def get_key_by_index(self, index: int) -> Any:
        self._build_index()
        return self._index_key[index]

    def get_value_by_index(self, index: int) -> Any:
        key = self.get_key_by_index(index)
        return self[key]

    def get_item_by_index(self, index: int) -> Tuple[Any, Any]:
        key = self.get_key_by_index(index)
        return key, self[key]

    def set_value_by_index(self, index: int, value: Any) -> None:
        key = self.get_key_by_index(index)
        self[key] = value

    def pop_by_index(self, index: int) -> Tuple[Any, Any]:
        key = self.get_key_by_index(index)
        return key, self.pop(key)

    def __setitem__(  # type: ignore
        self, key: Any, value: Any, *args: List[Any], **kwds: Dict[str, Any]
    ) -> None:
        self._need_reindex = key not in self
        super().__setitem__(key, value, *args, **kwds)  # type: ignore

    def __delitem__(  # type: ignore
        self, *args: List[Any], **kwds: Dict[str, Any]
    ) -> None:
        self._need_reindex = True
        super().__delitem__(*args, **kwds)  # type: ignore

    def clear(self) -> None:
        self._need_reindex = True
        super().clear()

    def copy(self) -> "IndexedOrderedDict":
        other = super().copy()
        assert isinstance(other, IndexedOrderedDict)
        other._need_reindex = self._need_reindex
        other._index_key = self._index_key.copy()
        other._key_index = self._key_index.copy()
        return other

    def popitem(  # type: ignore
        self, *args: List[Any], **kwds: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        self._need_reindex = True
        return super().popitem(*args, **kwds)  # type: ignore

    def move_to_end(  # type: ignore
        self, *args: List[Any], **kwds: Dict[str, Any]
    ) -> None:
        self._need_reindex = True
        super().move_to_end(*args, **kwds)  # type: ignore

    def __sizeof__(self) -> int:  # pragma: no cover
        return super().__sizeof__() + sys.getsizeof(self._need_reindex)

    def pop(  # type: ignore
        self, *args: List[Any], **kwds: Dict[str, Any]
    ) -> Any:
        self._need_reindex = True
        return super().pop(*args, **kwds)  # type: ignore

    def _build_index(self) -> None:
        if self._need_reindex:
            self._index_key = list(self.keys())
            self._key_index = {x: i for i, x in enumerate(self._index_key)}
            self._need_reindex = False


class ParamDict(IndexedOrderedDict):
    def __init__(self, data: Any = None, deep: bool = True):
        super().__init__()
        self.add(data, on_dup="overwrite", deep=deep)

    def __setitem__(  # type: ignore
        self, key: Any, value: Any, *args: List[Any], **kwds: Dict[str, Any]
    ) -> None:
        assert isinstance(key, str)
        super().__setitem__(key, value, *args, **kwds)  # type: ignore

    def get(self, key: str, default: Any) -> Any:  # type: ignore
        assert default is not None, "default value can't be None"
        if key in self:
            return as_type(self[key], type(default))
        return default

    def get_or_none(self, key: str, expected_type: type) -> Any:
        return self._get_or(key, expected_type, throw=False)

    def get_or_throw(self, key: str, expected_type: type) -> Any:
        return self._get_or(key, expected_type, throw=True)

    def _get_or(self, key: str, expected_type: type, throw: bool = True) -> Any:
        if key in self:
            return as_type(self[key], expected_type)
        if throw:
            raise KeyError(f"{key} not found")
        return None

    def to_json_str(self, indent: bool = False) -> str:
        if not indent:
            return json.dumps(self, separators=(",", ":"))
        else:
            return json.dumps(self, indent=4)

    def add(self, other: Any, on_dup: str, deep: bool = True) -> "ParamDict":
        on_dup = on_dup.lower()
        for k, v in to_kv_iterable(other):
            if on_dup == "overwrite" or k not in self:
                self[k] = copy.deepcopy(v) if deep else v
            elif on_dup == "throw":
                raise KeyError(f"{k} exists in dict")
            elif on_dup == "ignore":
                continue
            else:
                raise ValueError(f"{on_dup} is not supported")
        return self
