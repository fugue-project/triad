import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple


class IndexedOrderedDict(OrderedDict):
    def __init__(self, *args: List[Any], **kwds: Dict[str, Any]):
        super().__init__(*args, **kwds)
        self.__need_reindex = True
        self.__key_index: Dict[Any, int] = {}
        self.__index_key: List[Any] = []

    def index_of_key(self, key: Any) -> int:
        self._build_index()
        return self.__key_index[key]

    def get_key_by_index(self, index: int) -> Any:
        self._build_index()
        return self.__index_key[index]

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
        self.__need_reindex = key not in self
        super().__setitem__(key, value, *args, **kwds)  # type: ignore

    def __delitem__(  # type: ignore
        self, *args: List[Any], **kwds: Dict[str, Any]
    ) -> None:
        self.__need_reindex = True
        super().__delitem__(*args, **kwds)  # type: ignore

    def clear(self) -> None:
        self.__need_reindex = True
        super().clear()

    def popitem(  # type: ignore
        self, *args: List[Any], **kwds: Dict[str, Any]
    ) -> Tuple[Any, Any]:
        self.__need_reindex = True
        return super().popitem(*args, **kwds)  # type: ignore

    def move_to_end(  # type: ignore
        self, *args: List[Any], **kwds: Dict[str, Any]
    ) -> None:
        self.__need_reindex = True
        super().move_to_end(*args, **kwds)  # type: ignore

    def __sizeof__(self) -> int:
        return super().__sizeof__() + sys.getsizeof(self.__need_reindex)

    def pop(  # type: ignore
        self, *args: List[Any], **kwds: Dict[str, Any]
    ) -> Any:
        self.__need_reindex = True
        return super().pop(*args, **kwds)  # type: ignore

    def setdefault(  # type: ignore
        self, *args: List[Any], **kwds: Dict[str, Any]
    ) -> Any:
        self.__need_reindex = True
        return super().setdefault(*args, **kwds)  # type: ignore

    def _build_index(self) -> None:
        if self.__need_reindex:
            self.__index_key = list(self.keys())
            self.__key_index = {x: i for i, x in enumerate(self.__index_key)}
            self.__need_reindex = False
