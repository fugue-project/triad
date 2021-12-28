import copy
import json
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, TypeVar, Union

from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_arg_not_none
from triad.utils.convert import as_type
from triad.utils.iter import to_kv_iterable

KT = TypeVar("KT")
VT = TypeVar("VT")


class IndexedOrderedDict(OrderedDict, Dict[KT, VT]):
    """Subclass of OrderedDict that can get and set with index"""

    def __init__(self, *args: Any, **kwds: Any):
        self._readonly = False
        self._need_reindex = True
        self._key_index: Dict[Any, int] = {}
        self._index_key: List[Any] = []
        super().__init__(*args, **kwds)

    @property
    def readonly(self) -> bool:
        """Whether this dict is readonly"""
        return self._readonly

    def set_readonly(self) -> None:
        """Make this dict readonly"""
        self._readonly = True

    def index_of_key(self, key: Any) -> int:
        """Get index of key

        :param key: key value
        :return: index of the key value
        """
        self._build_index()
        return self._key_index[key]

    def get_key_by_index(self, index: int) -> KT:
        """Get key by index

        :param index: index of the key
        :return: key value at the index
        """
        self._build_index()
        return self._index_key[index]

    def get_value_by_index(self, index: int) -> VT:
        """Get value by index

        :param index: index of the item
        :return: value at the index
        """
        key = self.get_key_by_index(index)
        return self[key]

    def get_item_by_index(self, index: int) -> Tuple[KT, VT]:
        """Get key value pair by index

        :param index: index of the item
        :return: key value tuple at the index
        """
        key = self.get_key_by_index(index)
        return key, self[key]

    def set_value_by_index(self, index: int, value: VT) -> None:
        """Set value by index

        :param index: index of the item
        :param value: new value
        """
        key = self.get_key_by_index(index)
        self[key] = value

    def pop_by_index(self, index: int) -> Tuple[KT, VT]:
        """Pop item at index

        :param index: index of the item
        :return: key value tuple at the index
        """
        key = self.get_key_by_index(index)
        return key, self.pop(key)

    def equals(self, other: Any, with_order: bool):
        """Compare with another object

        :param other: for possible types, see :func:`~triad.utils.iter.to_kv_iterable`
        :param with_order: whether to compare order
        :return: whether they equal
        """
        if with_order:
            if isinstance(other, OrderedDict):
                return self == other
            return self == OrderedDict(to_kv_iterable(other))
        else:
            if isinstance(other, OrderedDict) or not isinstance(other, Dict):
                return self == dict(to_kv_iterable(other))
            return self == other

    # ----------------------------------- Wrappers over OrderedDict

    def __setitem__(  # type: ignore
        self, key: KT, value: VT, *args: Any, **kwds: Any
    ) -> None:
        self._pre_update("__setitem__", key not in self)
        super().__setitem__(key, value, *args, **kwds)  # type: ignore

    def __delitem__(self, *args: Any, **kwds: Any) -> None:  # type: ignore
        self._pre_update("__delitem__")
        super().__delitem__(*args, **kwds)  # type: ignore

    def clear(self) -> None:
        self._pre_update("clear")
        super().clear()

    def copy(self) -> "IndexedOrderedDict":
        other = super().copy()
        assert isinstance(other, IndexedOrderedDict)
        other._need_reindex = self._need_reindex
        other._index_key = self._index_key.copy()
        other._key_index = self._key_index.copy()
        other._readonly = False
        return other

    def __copy__(self) -> "IndexedOrderedDict":
        return self.copy()

    def __deepcopy__(self, arg: Any) -> "IndexedOrderedDict":
        it = [(copy.deepcopy(k), copy.deepcopy(v)) for k, v in self.items()]
        return IndexedOrderedDict(it)

    def popitem(self, *args: Any, **kwds: Any) -> Tuple[KT, VT]:  # type: ignore
        self._pre_update("popitem")
        return super().popitem(*args, **kwds)  # type: ignore

    def move_to_end(self, *args: Any, **kwds: Any) -> None:  # type: ignore
        self._pre_update("move_to_end")
        super().move_to_end(*args, **kwds)  # type: ignore

    def __sizeof__(self) -> int:  # pragma: no cover
        return super().__sizeof__() + sys.getsizeof(self._need_reindex)

    def pop(self, *args: Any, **kwds: Any) -> VT:  # type: ignore
        self._pre_update("pop")
        return super().pop(*args, **kwds)  # type: ignore

    def _build_index(self) -> None:
        if self._need_reindex:
            self._index_key = list(self.keys())
            self._key_index = {x: i for i, x in enumerate(self._index_key)}
            self._need_reindex = False

    def _pre_update(self, op: str, need_reindex: bool = True) -> None:
        if self.readonly:
            raise InvalidOperationError("This dict is readonly")
        self._need_reindex = need_reindex


class ParamDict(IndexedOrderedDict[str, Any]):
    """Parameter dictionary, a subclass of ``IndexedOrderedDict``, keys must be string

    :param data: for possible types, see :func:`~triad.utils.iter.to_kv_iterable`
    :param deep: whether to deep copy ``data``
    """

    OVERWRITE = 0
    THROW = 1
    IGNORE = 2

    def __init__(self, data: Any = None, deep: bool = True):
        super().__init__()
        self.update(data, deep=deep)

    def __setitem__(  # type: ignore
        self, key: str, value: Any, *args: Any, **kwds: Any
    ) -> None:
        assert isinstance(key, str)
        super().__setitem__(key, value, *args, **kwds)  # type: ignore

    def __getitem__(self, key: Union[str, int]) -> Any:  # type: ignore
        if isinstance(key, int):
            key = self.get_key_by_index(key)
        return super().__getitem__(key)  # type: ignore

    def get(self, key: Union[int, str], default: Any) -> Any:  # type: ignore
        """Get value by ``key``, and the value must be a subtype of the type of
        ``default``(which can't be None). If the ``key`` is not found,
        return ``default``.

        :param key: the key to search
        :raises NoneArgumentError: if default is None
        :raises TypeError: if the value can't be converted to the type of ``default``

        :return: the value by ``key``, and the value must be a subtype of the type of
            ``default``. If ``key`` is not found, return `default`
        """
        assert_arg_not_none(default, "default")
        if (isinstance(key, str) and key in self) or isinstance(key, int):
            return as_type(self[key], type(default))
        return default

    def get_or_none(self, key: Union[int, str], expected_type: type) -> Any:
        """Get value by `key`, and the value must be a subtype of ``expected_type``

        :param key: the key to search
        :param expected_type: expected return value type

        :raises TypeError: if the value can't be converted to ``expected_type``

        :return: if ``key`` is not found, None. Otherwise if the value can be converted
            to ``expected_type``, return the converted value, otherwise raise exception
        """
        return self._get_or(key, expected_type, throw=False)

    def get_or_throw(self, key: Union[int, str], expected_type: type) -> Any:
        """Get value by ``key``, and the value must be a subtype of ``expected_type``.
        If ``key`` is not found or value can't be converted to ``expected_type``, raise
        exception

        :param key: the key to search
        :param expected_type: expected return value type

        :raises KeyError: if ``key`` is not found
        :raises TypeError: if the value can't be converted to ``expected_type``

        :return: only when ``key`` is found and can be converted to ``expected_type``,
            return the converted value
        """
        return self._get_or(key, expected_type, throw=True)

    def to_json(self, indent: bool = False) -> str:
        """Generate json expression string for the dictionary

        :param indent: whether to have indent
        :return: json string
        """
        if not indent:
            return json.dumps(self, separators=(",", ":"))
        else:
            return json.dumps(self, indent=4)

    def update(  # type: ignore
        self, other: Any, on_dup: int = 0, deep: bool = True
    ) -> "ParamDict":
        """Update dictionary with another object (for possible types,
        see :func:`~triad.utils.iter.to_kv_iterable`)

        :param other: for possible types, see :func:`~triad.utils.iter.to_kv_iterable`
        :param on_dup: one of ``ParamDict.OVERWRITE``, ``ParamDict.THROW``
            and ``ParamDict.IGNORE``

        :raises KeyError: if using ``ParamDict.THROW`` and other contains existing keys
        :raises ValueError: if ``on_dup`` is invalid
        :return: itself
        """
        self._pre_update("update", True)
        for k, v in to_kv_iterable(other):
            if on_dup == ParamDict.OVERWRITE or k not in self:
                self[k] = copy.deepcopy(v) if deep else v
            elif on_dup == ParamDict.THROW:
                raise KeyError(f"{k} exists in dict")
            elif on_dup == ParamDict.IGNORE:
                continue
            else:
                raise ValueError(f"{on_dup} is not supported")
        return self

    def _get_or(
        self, key: Union[int, str], expected_type: type, throw: bool = True
    ) -> Any:
        if (isinstance(key, str) and key in self) or isinstance(key, int):
            return as_type(self[key], expected_type)
        if throw:
            raise KeyError(f"{key} not found")
        return None
