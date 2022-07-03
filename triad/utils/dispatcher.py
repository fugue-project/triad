import itertools
import logging
from functools import update_wrapper
from typing import Any, Callable, Iterable, List, Optional, Tuple

try:
    from importlib.metadata import entry_points  # type:ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import entry_points  # type:ignore


def run_at_def(run_at_def_func: Optional[Callable] = None, **kwargs: Any) -> Callable:
    def _run(_func: Callable) -> Callable:
        _func(**kwargs)
        return _func

    return _run if run_at_def_func is None else _run(run_at_def_func)  # type:ignore


def conditional_dispatcher(
    default_func: Optional[Callable[..., Any]] = None, entry_point: Optional[str] = None
) -> Callable:
    def _run(_func: Callable) -> "ConditionalDispatcher":
        class _Dispatcher(ConditionalDispatcher):
            def __call__(self, *args: Any, **kwds: Any) -> Any:
                return self.run_top(*args, **kwds)

        return _Dispatcher(_func, entry_point=entry_point)

    return _run if default_func is None else _run(default_func)  # type:ignore


def conditional_broadcaster(
    default_func: Optional[Callable[..., Any]] = None, entry_point: Optional[str] = None
) -> Callable:
    def _run(_func: Callable) -> "ConditionalDispatcher":
        class _Dispatcher(ConditionalDispatcher):
            def __call__(self, *args: Any, **kwds: Any) -> None:
                list(self.run(*args, **kwds))

        return _Dispatcher(_func, entry_point=entry_point)

    return _run if default_func is None else _run(default_func)  # type:ignore


class ConditionalDispatcher:
    def __init__(
        self, default_func: Callable[..., Any], entry_point: Optional[str] = None
    ):
        self._func = default_func
        self._funcs: List[
            Tuple[float, int, Callable[..., bool], Callable[..., Any]]
        ] = []
        self._entry_point = entry_point
        self._preloaded = False
        update_wrapper(self, default_func)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError  # pragma: no cover

    def run(self, *args: Any, **kwargs: Any) -> Iterable[Any]:
        self._preload()
        has_return = False
        for f in self._funcs:
            if self._match(f[2], *args, **kwargs):
                yield f[3](*args, **kwargs)
                has_return = True
        if not has_return:
            yield self._func(*args, **kwargs)

    def run_top(self, *args: Any, **kwargs: Any) -> Any:
        return list(itertools.islice(self.run(*args, **kwargs), 1))[0]

    def register(
        self,
        func: Callable[..., Any],
        matcher: Callable[..., bool],
        priority: float = 1.0,
    ) -> None:
        n = len(self._funcs)
        self._funcs.append((-priority, -n, matcher, func))
        self._funcs.sort()

    def candidate(
        self,
        matcher: Callable[..., bool],
        priority: float = 1.0,
    ) -> Callable:
        def _run(_func: Callable[..., Any]):
            self.register(_func, matcher=matcher, priority=priority)
            return _func

        return _run

    def _preload(self) -> None:
        if self._entry_point is not None and not self._preloaded:
            logger = logging.getLogger(self._func.__name__)
            for plugin in _entry_points().get(self._entry_point, []):
                try:
                    plugin.load()
                    logger.debug("loaded %s", plugin)
                except Exception as e:  # pragma: no cover
                    logger.debug("failed to load %s: %s", plugin, e)
            self._preloaded = True

    def _match(self, m: Callable[..., bool], *args: Any, **kwargs: Any) -> bool:
        try:
            return m(*args, **kwargs)
        except Exception:
            return False


def _entry_points() -> Any:
    return entry_points()
