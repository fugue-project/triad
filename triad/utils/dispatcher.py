import itertools
from functools import update_wrapper
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .entry_points import load_entry_point


def run_at_def(run_at_def_func: Optional[Callable] = None, **kwargs: Any) -> Callable:
    """Decorator to run the function at declaration. This is useful when we want import
    to trigger a function run (which can guarantee it runs only once).

    .. admonition:: Examples

        Assume the following python file is a module in your package,
        then when you ``import package.module``, the two functions will run.

        .. code-block:: python

            from triad import run_at_def

            @run_at_def
            def register_something():
                print("registered")

            @run_at_def(a=1)
            def register_something2(a):
                print("registered", a)

    :param run_at_def_func: the function to decorate
    :param kwargs: the parameters to call this function
    """

    def _run(_func: Callable) -> Callable:
        _func(**kwargs)
        return _func

    return _run if run_at_def_func is None else _run(run_at_def_func)  # type:ignore


def conditional_dispatcher(
    default_func: Optional[Callable[..., Any]] = None, entry_point: Optional[str] = None
) -> Callable:
    """Decorating a conditional dispatcher that will run the **first matching** registered
    functions in other modules/packages. This is a more general solution compared to
    ``functools.singledispatch``. You can write arbitrary matching functions according
    to all the inputs of the function.

    .. admonition:: Examples

        Assume in ``pkg1.module1``, you have:

        .. code-block:: python

            from triad import conditional_dispatcher

            @conditional_dispatcher(entry_point="my.plugins")
            def get_len(obj):
                raise NotImplementedError

        In another package ``pkg2``, in ``setup.py``, you define
        an entry point as:

        .. code-block:: python

            setup(
                ...,
                entry_points={
                    "my.plugins": [
                        "my = pkg2.module2"
                    ]
                },
            )

        And in ``pkg2.module2``:

        .. code-block:: python

            from pkg1.module1 import get_len

            @get_len.candidate(lambda obj: isinstance(obj, str))
            def get_str_len(obj:str) -> int:
                return len(obj)

            @get_len.candidate(lambda obj: isinstance(obj, int) and obj == 10)
            def get_int_len(obj:int) -> int:
                return obj

        Now, both functions will be automatically registered when ``pkg2``
        is installed in the environement. In another ``pkg3``:

        .. code-block:: python

            from pkg1.module1 import get_len

            assert get_len("abc") == 3  # calling get_str_len
            assert get_len(10) == 10  # calling get_int_len
            get_len(20)  # raise NotImplementedError due to no matching candidates

    .. seealso::

        Please read :meth:`~.ConditionalDispatcher.candidate` for details about the
        matching function and priority settings.

    :param default_func: the function to decorate
    :param entry_point: the entry point to preload dispatchers, defaults to None
    """

    def _run(_func: Callable) -> "ConditionalDispatcher":
        class _Dispatcher(ConditionalDispatcher):
            def __call__(self, *args: Any, **kwds: Any) -> Any:
                return self.run_top(*args, **kwds)

        return _Dispatcher(_func, entry_point=entry_point)

    return _run if default_func is None else _run(default_func)  # type:ignore


def conditional_broadcaster(
    default_func: Optional[Callable[..., Any]] = None, entry_point: Optional[str] = None
) -> Callable:
    """Decorating a conditional broadcaster that will run **all** registered functions in
    other modules/packages.

    .. admonition:: Examples

        Assume in ``pkg1.module1``, you have:

        .. code-block:: python

            from triad import conditional_broadcaster

            @conditional_broadcaster(entry_point="my.plugins")
            def myprint(obj):
                raise NotImplementedError

            @conditional_broadcaster(entry_point="my.plugins")
            def myprint2(obj):
                raise NotImplementedError

        In another package ``pkg2``, in ``setup.py``, you define
        an entry point as:

        .. code-block:: python

            setup(
                ...,
                entry_points={
                    "my.plugins": [
                        "my = pkg2.module2"
                    ]
                },
            )

        And in ``pkg2.module2``:

        .. code-block:: python

            from pkg1.module1 import get_len

            @myprint.candidate(lambda obj: isinstance(obj, str))
            def myprinta(obj:str) -> None:
                print(obj, "a")

            @myprint.candidate(lambda obj: isinstance(obj, str) and obj == "x")
            def myprintb(obj:str) -> None:
                print(obj, "b")

        Now, both functions will be automatically registered when ``pkg2``
        is installed in the environement. In another ``pkg3``:

        .. code-block:: python

            from pkg1.module1 import get_len

            myprint("x")  # calling both myprinta and myprinta
            myprint("y")  # calling myprinta only
            myprint2("x")  # raise NotImplementedError due to no matching candidates

    .. note::

        Only when no matching candidate found, the implementation of the original
        function will be used. If you don't want to throw an error, then use ``pass`` in
        the original function instead.

    .. seealso::

        Please read :meth:`~.ConditionalDispatcher.candidate` for details about the
        matching function and priority settings.

    :param default_func: the function to decorate
    :param entry_point: the entry point to preload dispatchers, defaults to None
    """

    def _run(_func: Callable) -> "ConditionalDispatcher":
        class _Dispatcher(ConditionalDispatcher):
            def __call__(self, *args: Any, **kwds: Any) -> None:
                list(self.run(*args, **kwds))

        return _Dispatcher(_func, entry_point=entry_point)

    return _run if default_func is None else _run(default_func)  # type:ignore


class ConditionalDispatcher:
    """A conditional function dispatcher based on custom matching functions.
    This is a more general solution compared to
    ``functools.singledispatch``. You can write arbitrary matching functions according
    to all the inputs of the function.

    .. note::

        Please use the decorators :func:`.conditional_dispatcher` and
        :func:`.conditional_broadcaster` instead of directly using this class.

    :param default_func: the parent function that will dispatch the execution
        based on matching functions
    :param entry_point: the entry point to preload children functions,
        defaults to None
    """

    def __init__(
        self, default_func: Callable[..., Any], entry_point: Optional[str] = None
    ):
        self._func = default_func
        self._funcs: List[
            Tuple[float, int, Callable[..., bool], Callable[..., Any]]
        ] = []
        self._entry_point = entry_point
        update_wrapper(self, default_func)

    def __getstate__(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if k in ["_func", "_funcs", "_entry_point"]
        }

    def __setstate__(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            setattr(self, k, v)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """The abstract method to mimic the function call"""
        raise NotImplementedError  # pragma: no cover

    def run(self, *args: Any, **kwargs: Any) -> Iterable[Any]:
        """Execute all matching children functions as a generator.

        .. note::

            Only when there is matching functions, the default implementation
            will be invoked.
        """
        self._preload()
        has_return = False
        for f in self._funcs:
            if self._match(f[2], *args, **kwargs):
                yield f[3](*args, **kwargs)
                has_return = True
        if not has_return:
            yield self._func(*args, **kwargs)

    def run_top(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the first matching child function

        :return: the return of the child function
        """
        return list(itertools.islice(self.run(*args, **kwargs), 1))[0]

    def register(
        self,
        func: Callable[..., Any],
        matcher: Callable[..., bool],
        priority: float = 1.0,
    ) -> None:
        """Register a child function with matcher and priority.

        .. note::

            The order to be matched is determined by both the priority
            and the order of registration.

            * The default priority is 1.0
            * Children with higher priority values will be matched earlier
            * When ``priority>0`` then later registrations will be matched earlier
            * When ``priority<=0`` then earlier registrations will be matched earlier

            So if you want to 'overwrite' the existed matches, set priority to be
            greater than 1.0. If you want to 'ignore' the current if there are other
            matches, set priority to 0.0.

        :param func: a child function to be used when matching
        :param matcher: a function determines whether it is a match
            based on the same input as the parent function
        :param priority: it determines the order to be matched,
            **higher value means higher priority**, defaults to 1.0
        """
        n = len(self._funcs)
        self._funcs.append((-priority, n if priority <= 0.0 else -n, matcher, func))
        self._funcs.sort()

    def candidate(
        self,
        matcher: Callable[..., bool],
        priority: float = 1.0,
    ) -> Callable:
        """A decorator to register a child function with matcher and priority.

        .. note::

            The order to be matched is determined by both the priority
            and the order of registration.

            * The default priority is 1.0
            * Children with higher priority values will be matched earlier
            * When ``priority>0`` then later registrations will be matched earlier
            * When ``priority<=0`` then earlier registrations will be matched earlier

            So if you want to 'overwrite' the existed matches, set priority to be
            greater than 1.0. If you want to 'ignore' the current if there are other
            matches, set priority to 0.0.

        .. seealso::

            Please see examples in :func:`.conditional_dispatcher` and
            :func:`.conditional_broadcaster`.

        :param matcher: a function determines whether it is a match
            based on the same input as the parent function
        :param priority: it determines the order to be matched,
            **higher value means higher priority**, defaults to 1.0
        """

        def _run(_func: Callable[..., Any]):
            self.register(_func, matcher=matcher, priority=priority)
            return _func

        return _run

    def _preload(self) -> None:
        if self._entry_point is not None:
            load_entry_point(self._entry_point)

    def _match(self, m: Callable[..., bool], *args: Any, **kwargs: Any) -> bool:
        try:
            return m(*args, **kwargs)
        except Exception:
            return False
