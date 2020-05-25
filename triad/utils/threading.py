from threading import RLock
from typing import Any, Callable, Dict, Tuple


class RunOnce(object):
    """Run `func` once, the uniqueness is defined by `key_func`

    :param func: the function to run only once with this wrapper instance
    :param key_func: the unique key determined by arguments of `func`
    :param lock_type: lock class type for thread safe

    :Examples:
    >>> r = RunOnce(max, lambda *args, **kwargs: id(args[0]))
    >>> a1 = [0, 1]
    >>> a2 = [0, 1]
    >>> assert 1 == r(a1) # will trigger max
    >>> assert 1 == r(a1) # will get the result from cache
    >>> assert 1 == r(a2) # will trigger max again because id(a1)!=id(a2)

    :Notice:
    * Hash collision is the concern of the user, not this class, your
      `key_func` should avoid any potential collision
    * `func` can have no return
    * For concurrent calls of this wrapper, only one will trigger `func` other
      calls will be blocked until the first call returns an result
    * This class is cloudpicklable, but unpickled instance does NOT share the same
      context with the original one
    """

    def __init__(self, func: Callable, key_func: Callable, lock_type: type = RLock):
        self._func = func
        self._key_func = key_func
        self._lock_type = lock_type
        self._init_locks()

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        del d["_lock"]
        del d["_locks"]
        del d["_store"]
        return d

    def __setstate__(self, members: Any) -> None:
        self.__dict__.update(members)
        self._init_locks()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        key = self._key_func(*args, **kwargs)
        lock = self._get_lock(key)
        with lock:
            found, res = self._try_get(key)
            if found:
                return res
            res = self._func(*args, **kwargs)
            self._update(key, res)
            return res

    def _get_lock(self, key) -> Any:
        with self._lock:
            if key not in self._locks:
                self._locks[key] = self._lock_type()
            return self._locks[key]

    def _try_get(self, key) -> Tuple[bool, Any]:
        with self._lock:
            if key in self._store:
                return True, self._store[key]
            return False, None

    def _update(self, key, value) -> None:
        with self._lock:
            self._store[key] = value

    def _init_locks(self):
        self._lock = self._lock_type()
        self._locks: Dict[int, Any] = {}
        self._store: Dict[int, Any] = {}
