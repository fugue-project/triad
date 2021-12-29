from typing import Callable, Dict, Optional, Type, get_type_hints, Any
import inspect
from triad.utils.assertion import assert_or_throw


def _get_first_arg_type(func: Callable) -> Any:
    sig = inspect.signature(func)
    annotations = get_type_hints(func)
    for k, w in sig.parameters.items():
        assert_or_throw(k != "self", ValueError(f"class method is not allowed {func}"))
        assert_or_throw(
            w.kind == w.POSITIONAL_OR_KEYWORD,
            ValueError(f"{w} is not a valid parameter in {func}"),
        )
        anno = annotations.get(k, w.annotation)
        assert_or_throw(
            anno != inspect.Parameter.empty,
            ValueError(f"the first argument must be annotated in {func}"),
        )
        return anno
    raise ValueError(f"{func} does not have any input parameter")


class _ClassExtension:
    def __init__(self, class_type: Type):
        self._class_type = class_type
        self._built_in = set(dir(class_type))
        self._ext: Dict[str, Callable] = {}

    def add_method(
        self, func: Callable, name: Optional[str] = None, on_dup: str = "error"
    ) -> None:
        assert_or_throw(
            name not in self._built_in, ValueError(f"{name} is a built in attribute")
        )
        if name is None:
            name = func.__name__
        if name in self._ext:
            if on_dup == "ignore":
                return
            if on_dup == "error":
                raise ValueError(f"{name} is already registered")
        self._ext[name] = func
        setattr(self._class_type, name, func)


class _ClassExtensions:
    def __init__(self):
        self._types: Dict[Type, _ClassExtension] = {}

    def register_type(self, tp: Type) -> None:
        assert_or_throw(
            tp not in self._types, ValueError(f"{tp} is already registered")
        )
        self._types[tp] = _ClassExtension(tp)

    def add_method(
        self,
        class_type: Type,
        func: Callable,
        name: Optional[str] = None,
        on_dup: str = "error",
    ) -> None:
        assert_or_throw(
            class_type in self._types, ValueError(f"{class_type} is not registered")
        )
        self._types[class_type].add_method(func, name=name, on_dup=on_dup)


_CLASS_EXTENSIONS = _ClassExtensions()


def extensible_class(class_type: Type) -> Type:
    """The decorator making classes extensible by external methods

    :param class_type: the class under the decorator
    :return: the ``class_type``

    .. admonition:: Examples

        .. code-block:: python

            @extensible_class
            class A:

                # It's recommended to implement __getattr__ so that
                # PyLint will not complain about the dynamically added methods
                def __getattr__(self, name):
                    raise NotImplementedError

            @extension_method
            def method(obj:A):
                return 1

            assert 1 == A().method()

    .. note::

        If the method name is already in the original class, a ValueError will be
        thrown. You can't modify any built-in attribute.
    """
    _CLASS_EXTENSIONS.register_type(class_type)
    return class_type


def extension_method(
    func: Optional[Callable] = None,
    class_type: Optional[Type] = None,
    name: Optional[str] = None,
    on_dup: str = "error",
) -> Callable:
    """The decorator to add functions as members of the
    correspondent classes.

    :param func: the function under the decorator
    :param class_type: the parent class type, defaults to None
    :param name: the specified class method name, defaults to None. If None
      then ``func.__name__`` will be used as the method name
    :param on_dup: action on name duplication, defaults to ``error``. ``error``
      will throw a ValueError; ``ignore`` will take no action; ``overwrite``
      will use the current method to overwrite.
    :return: the underlying function

    .. admonition:: Examples

        .. code-block:: python

            @extensible_class
            class A:

                # It's recommended to implement __getattr__ so that
                # PyLint will not complain about the dynamically added methods
                def __getattr__(self, name):
                    raise NotImplementedError

            # The simplest way to use this decorator, the first argument of
            # the method must be annotated, and the annotated type is the
            # class type to add this method to.
            @extension_method
            def method1(obj:A):
                return 1

            assert 1 == A().method1()

            # Or you can be explicit of the class type and the name of the
            # method in the class. In this case, you don't have to annotate
            # the first argument.
            @extension_method(class_type=A, name="m3")
            def method2(obj, b):
                return 2 + b

            assert 5 == A().m3(3)

    .. note::

        If the method name is already in the original class, a ValueError will be
        thrown. You can't modify any built-in attribute.
    """
    if func is not None:  # @extension_method
        _CLASS_EXTENSIONS.add_method(
            _get_first_arg_type(func) if class_type is None else class_type,
            func=func,
            name=name,
            on_dup=on_dup,
        )
        return func
    else:  # @extension_method(...)

        def inner(func):
            _CLASS_EXTENSIONS.add_method(
                _get_first_arg_type(func) if class_type is None else class_type,
                func=func,
                name=name,
                on_dup=on_dup,
            )
            return func

        return inner
