from functools import _make_key, wraps
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple, Type


class SerializableRLock:
    """A serialization safe wrapper of :external+python:class:`threading.RLock`"""

    def __init__(self):
        self._lock = RLock()

    def __enter__(self) -> Any:
        return self._lock.__enter__()

    def __exit__(
        self, exception_type: Any, exception_value: Any, exception_traceback: Any
    ) -> Any:
        return self._lock.__exit__(exception_type, exception_value, exception_traceback)

    def __getstate__(self) -> Dict[str, Any]:
        return {}

    def __setstate__(self, data: Dict[str, Any]) -> None:
        self._lock = RLock()


def run_once(
    func: Optional[Callable] = None,
    key_func: Optional[Callable] = None,
    lock_type: Type = RLock,
) -> Callable:
    """The decorator to run `func` once, the uniqueness is defined by `key_func`.
    This implementation is serialization safe and thread safe.

    :param func: the function to run only once with this wrapper instance
    :param key_func: the unique key determined by arguments of `func`, if not set, it
      will use the same hasing logic as :external+python:func:`functools.lru_cache`
    :param lock_type: lock class type for thread safe, it doesn't need to be
      serialization safe

    .. admonition:: Examples

        .. code-block:: python

            @run_once
            def r(a):
                return max(a)

            a1 = [0, 1]
            a2 = [0, 2]
            assert 1 == r(a1) # will trigger r
            assert 1 == r(a1) # will get the result from cache
            assert 2 == r(a2) # will trigger r again because of different arguments

            # the following example ignores arguments
            @run_once(key_func=lambda *args, **kwargs: True)
            def r2(a):
                return max(a)

            assert 1 == r(a1) # will trigger r
            assert 1 == r(a2) # will get the result from cache

    .. note::

        * Hash collision is the concern of the user, not this class, your
          `key_func` should avoid any potential collision
        * `func` can have no return
        * For concurrent calls of this wrapper, only one will trigger `func` other
          calls will be blocked until the first call returns an result
        * This class is cloudpicklable, but unpickled instance does NOT share the same
          context with the original one
        * This is not to replace :external+python:func:`functools.lru_cache`,
          it is not supposed to cache a lot of items
    """

    def _run(func: Callable) -> "RunOnce":
        return wraps(func)(RunOnce(func, key_func=key_func, lock_type=lock_type))

    return _run(func) if func is not None else wraps(func)(_run)  # type: ignore


class RunOnce:
    """Run `func` once, the uniqueness is defined by `key_func`.
    This implementation is serialization safe and thread safe.

    .. note::

        Please use the decorator :func:`~.run_once` instead of directly
        using this class

    :param func: the function to run only once with this wrapper instance
    :param key_func: the unique key determined by arguments of `func`, if not set, it
        will use the same hasing logic as :external+python:func:`functools.lru_cache`
    :param lock_type: lock class type for thread safe
    """

    def __init__(
        self,
        func: Callable,
        key_func: Optional[Callable] = None,
        lock_type: Type = RLock,
    ):
        self._func = func
        if key_func is None:
            self._key_func: Callable = lambda *args, **kwargs: _make_key(
                args, kwargs, typed=True
            )
        else:
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
