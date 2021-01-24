import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import six
from ciso8601 import parse_datetime
from triad.utils.assertion import assert_or_throw
from triad.utils.string import assert_triad_var_name

EMPTY_ARGS: List[Any] = []
EMPTY_KWARGS: Dict[str, Any] = {}


def get_caller_global_local_vars(
    global_vars: Optional[Dict[str, Any]] = None,
    local_vars: Optional[Dict[str, Any]] = None,
    start: int = -1,
    end: int = -1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Get the caller level global and local variables.

    :param global_vars: overriding global variables, if not None,
      will return this instead of the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if not None,
      will return this instead of the caller's locals(), defaults to None
    :param start: start stack level (from 0 to any negative number),
      defaults to -1 which is one level above where this function is invoked
    :param end: end stack level (from ``start`` to any smaller negative number),
      defaults to -1 which is one level above where this function is invoked
    :return: tuple of `global_vars` and `local_vars`

    :Examples:

        .. code-block:: python
            def caller():
                x=1
                assert 1 == get_value("x")

            def get_value(var_name):
                _, l = get_caller_global_local_vars()
                assert var_name in l
                assert var_name not in locals()
                return l[var_name]

    :Notice:

    This is for internal use, users normally should not call this directly.

    If merging multiple levels, the variables on closer level
    (to where it is invoked) will overwrite the further levels values if there
    is overlap.

    :Examples:

        .. code-block:: python
            def f1():
                x=1

                def f2():
                    x=2

                    def f3():
                        _, l = get_caller_global_local_vars(start=-1,end=-2)
                        assert 2 == l["x"]

                        _, l = get_caller_global_local_vars(start=-2,end=-2)
                        assert 1 == l["x"]

                f2()
            f1()
    """
    assert_or_throw(start <= 0, ValueError(f"{start} > 0"))
    assert_or_throw(end <= start, ValueError(f"{end} > {start}"))
    stack = inspect.currentframe().f_back  # type: ignore
    p = 0
    while p > start and stack is not None:
        stack = stack.f_back
        p -= 1
    g_arr: List[Dict[str, Any]] = []
    l_arr: List[Dict[str, Any]] = []
    while p >= end and stack is not None:
        g_arr.insert(0, stack.f_globals)
        l_arr.insert(0, stack.f_locals)
        stack = stack.f_back
        p -= 1
    if global_vars is None:
        global_vars = {}
        for d in g_arr:
            global_vars.update(d)
    if local_vars is None:
        local_vars = {}
        for d in l_arr:
            local_vars.update(d)
    return global_vars, local_vars  # type: ignore


def str_to_object(
    expr: str,
    global_vars: Optional[Dict[str, Any]] = None,
    local_vars: Optional[Dict[str, Any]] = None,
) -> Any:
    """Convert string expression to object. The string expression must express
    a type with relative or full path, or express a local or global instance without
    brackets or operators.

    :param expr: string expression, see examples below
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None
    :return: the

    :raises ValueError: unable to find a matching object

    :Examples:

        .. code-block:: python
            class _Mock(object):
                def __init__(self, x=1):
                    self.x = x

            m = _Mock()
            assert 1 == str_to_object("m.x")
            assert 1 == str_to_object("m2.x", local_vars={"m2": m})
            assert RuntimeError == str_to_object("RuntimeError")
            assert _Mock == str_to_object("_Mock")

    :Notice:

    This function is to dynamically load an object from string expression.
    If you write that string expression as python code at the same location, it
    should generate the same result.
    """
    try:
        for p in expr.split("."):
            assert_triad_var_name(p)
        _globals, _locals = get_caller_global_local_vars(global_vars, local_vars)
        if "." not in expr:
            return eval(expr, _globals, _locals)
        parts = expr.split(".")
        v = _locals.get(parts[0], _globals.get(parts[0], None))
        if v is not None and not isinstance(v, ModuleType):
            return eval(expr, _globals, _locals)
        root = ".".join(parts[:-1])
        return getattr(importlib.import_module(root), parts[-1])
    except ValueError:
        raise  # pragma: no cover
    except Exception:
        raise ValueError(expr)


def str_to_type(
    s: str,
    expected_base_type: Optional[type] = None,
    global_vars: Optional[Dict[str, Any]] = None,
    local_vars: Optional[Dict[str, Any]] = None,
) -> type:
    """Given a string expression, find the first/last type from all import libraries.
    If the expression contains `.`, it's supposed to be a relative or full path of
    the type including modules.

    :param s: type expression, for example `triad.utils.iter.Slicer` or `str`
    :param expected_base_type: base class type that must satisfy, defaults to None
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None

    :raises TypeError: unable to find a matching type

    :return: found type
    """
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    try:
        obj = str_to_object(s, global_vars, local_vars)
    except ValueError:
        raise TypeError(f"{s} is not a type")
    assert_or_throw(isinstance(obj, type), TypeError(f"{obj} is not a type"))
    assert_or_throw(
        expected_base_type is None or issubclass(obj, expected_base_type),
        TypeError(f"{obj} is not a subtype of {expected_base_type}"),
    )
    return obj


def str_to_instance(
    s: str,
    expected_base_type: Optional[type] = None,
    args: List[Any] = EMPTY_ARGS,
    kwargs: Dict[str, Any] = EMPTY_KWARGS,
    global_vars: Optional[Dict[str, Any]] = None,
    local_vars: Optional[Dict[str, Any]] = None,
) -> Any:
    """Use :func:'~triad.utils.convert.str_to_type' to find a matching type
    and instantiate

    :param s: see :func:'~triad.utils.convert.str_to_type'
    :param expected_base_type: see :func:'~triad.utils.convert.str_to_type'
    :param args: args to instantiate the type
    :param kwargs: kwargs to instantiate the type
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None

    :return: the instantiated the object
    """
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    t = str_to_type(s, expected_base_type, global_vars, local_vars)
    return t(*args, **kwargs)


def to_type(
    s: Any,
    expected_base_type: Optional[type] = None,
    global_vars: Optional[Dict[str, Any]] = None,
    local_vars: Optional[Dict[str, Any]] = None,
) -> type:
    """Convert an object `s` to `type`
    * if `s` is `str`: see :func:'~triad.utils.convert.str_to_type'
    * if `s` is `type`: check `expected_base_type` and return itself
    * else: check `expected_base_type` and return itself

    :param s: see :func:'~triad.utils.convert.str_to_type'
    :param expected_base_type: see :func:'~triad.utils.convert.str_to_type'
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None

    :raises TypeError: if no matching type found

    :return: the matching type
    """
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    if isinstance(s, str):
        return str_to_type(s, expected_base_type, global_vars, local_vars)
    if isinstance(s, type):
        if expected_base_type is None or issubclass(s, expected_base_type):
            return s
        raise TypeError(f"Type mismatch {s} expected {expected_base_type}")
    t = type(s)
    if expected_base_type is None or issubclass(t, expected_base_type):
        return t
    raise TypeError(f"Type mismatch {s} expected {expected_base_type}")


def to_instance(
    s: Any,
    expected_base_type: Optional[type] = None,
    args: List[Any] = EMPTY_ARGS,
    kwargs: Dict[str, Any] = EMPTY_KWARGS,
    global_vars: Optional[Dict[str, Any]] = None,
    local_vars: Optional[Dict[str, Any]] = None,
) -> Any:
    """If s is str or type, then use :func:'~triad.utils.convert.to_type' to find
    matching type and instantiate. Otherwise return s if it matches constraints

    :param s: see :func:'~triad.utils.convert.to_type'
    :param expected_base_type: see :func:'~triad.utils.convert.to_type'
    :param args: args to instantiate the type
    :param kwargs: kwargs to instantiate the type
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None

    :raises ValueError: if s is an instance but not a (sub)type of `expected_base_type`
    :raises TypeError: if s is an instance, args and kwargs must be empty

    :return: the instantiated object
    """
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    if s is None:
        raise ValueError("None can't be converted to instance")
    if isinstance(s, (str, type)):
        t = to_type(s, expected_base_type, global_vars, local_vars)
        return t(*args, **kwargs)
    else:
        if expected_base_type is not None and not isinstance(s, expected_base_type):
            raise TypeError(f"{str(s)} is not a subclass of {str(expected_base_type)}")
        if len(args) > 0 or len(kwargs) > 0:
            raise ValueError(f"Can't instantiate {str(s)} with different parameters")
        return s


def to_function(
    func: Any,
    global_vars: Optional[Dict[str, Any]] = None,
    local_vars: Optional[Dict[str, Any]] = None,
) -> Any:  # noqa: C901
    """For an expression, it tries to find the matching function.

    :params s: a string expression or a callable
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None

    :raises AttributeError: if unable to find such a function

    :return: the matching function
    """
    if isinstance(func, str):
        global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
        try:
            func = str_to_object(func, global_vars, local_vars)
        except ValueError:
            raise AttributeError(f"{func} is not a function")
    assert_or_throw(
        callable(func) and not isinstance(func, six.class_types),
        AttributeError(f"{func} is not a function"),
    )
    return func


def get_full_type_path(obj: Any) -> str:
    """Get the full module path of the type (if `obj` is class or function) or type
    of the instance (if `obj` is an object instance)

    :param obj: a class/function type or an object instance
    :raises TypeError: if `obj` is None, lambda, or neither a class or a function
    :return: full path string
    """
    if obj is not None:
        if inspect.isclass(obj):
            return "{}.{}".format(obj.__module__, obj.__name__)
        if inspect.isfunction(obj):
            if obj.__name__.startswith("<lambda"):
                raise TypeError("Can't get full path for lambda functions")
            return "{}.{}".format(obj.__module__, obj.__name__)
        if isinstance(obj, object):
            return "{}.{}".format(obj.__class__.__module__, obj.__class__.__name__)
    raise TypeError(f"Unable to get type full path from {obj}")


def to_bool(obj: Any) -> bool:
    """Convert an object to python bool value. It can handle values
    like `True`, `true`, `yes`, `1`, etc

    :param obj: object
    :raises TypeError: if failed to convert
    :return: bool value
    """
    if obj is None:
        raise TypeError("None can't convert to bool")
    o = str(obj).lower()
    if o in ["true", "yes", "1"]:
        return True
    if o in ["false", "no", "0"]:
        return False
    raise TypeError(f"{o} can't convert to bool")


def to_datetime(obj: Any) -> datetime.datetime:
    """Convert an object to python datetime. If the object is a
    string, it will use `ciso8601.parse_datetime` to parse

    :param obj: object
    :raises TypeError: if failed to convert
    :return: datetime value
    """
    if obj is None:
        raise TypeError("None can't convert to datetime")
    if isinstance(obj, datetime.datetime):
        return obj
    if isinstance(obj, datetime.date):
        return datetime.datetime(obj.year, obj.month, obj.day)
    if isinstance(obj, str):
        try:
            return parse_datetime(obj)
        except Exception as e:
            raise TypeError(f"{obj} can't convert to datetime", e)
    raise TypeError(f"{type(obj)} {obj} can't convert to datetime")


def to_timedelta(obj: Any) -> datetime.timedelta:
    """Convert an object to python datetime.

    If the object is a string, `min` or `-inf` will return `timedelta.min`,
    `max` or `inf` will return `timedelta.max`; if the object is a number,
    the number will be used as the seconds argument; Otherwise it will use
    `pandas.to_timedelta` to parse the object.

    :param obj: object
    :raises TypeError: if failed to convert
    :return: timedelta value
    """
    if obj is None:
        raise TypeError("None can't convert to timedelta")
    if isinstance(obj, datetime.timedelta):
        return obj
    if np.isreal(obj):
        return datetime.timedelta(seconds=float(obj))
    try:
        return pd.to_timedelta(obj).to_pytimedelta()
    except Exception as e:
        if isinstance(obj, str):
            obj = obj.lower()
            if obj in ["min", "-inf"]:
                return datetime.timedelta.min
            elif obj in ["max", "inf"]:
                return datetime.timedelta.max
        raise TypeError(f"{type(obj)} {obj} can't convert to timedelta", e)


def as_type(obj: Any, target: type) -> Any:
    """Convert `obj` into `target` type

    :param obj: input object
    :param target: target type

    :return: object in the target type
    """
    if issubclass(type(obj), target):
        return obj
    if target == bool:
        return to_bool(obj)
    if target == datetime.datetime:
        return to_datetime(obj)
    if target == datetime.timedelta:
        return to_timedelta(obj)
    return target(obj)


def to_size(exp: Any) -> int:
    """Convert input value or expression to size
    For expression string, it must be in the format of
    `<value>` or `<value><unit>`. Value must be 0 or positive,
    default unit is byte if not provided. Unit can be `b`, `byte`,
    `k`, `kb`, `m`, `mb`, `g`, `gb`, `t`, `tb`.

    Args:
        exp (Any): expression string or numerical value

    Raises:
        ValueError: for invalid expression
        ValueError: for negative values

    Returns:
        int: size in byte
    """
    n, u = _parse_value_and_unit(exp)
    assert n >= 0.0, "Size can't be negative"
    if u in ["", "b", "byte", "bytes"]:
        return int(n)
    if u in ["k", "kb"]:
        return int(n * 1024)
    if u in ["m", "mb"]:
        return int(n * 1024 * 1024)
    if u in ["g", "gb"]:
        return int(n * 1024 * 1024 * 1024)
    if u in ["t", "tb"]:
        return int(n * 1024 * 1024 * 1024 * 1024)
    raise ValueError(f"Invalid size expression {exp}")


def _parse_value_and_unit(exp: Any) -> Tuple[float, str]:
    try:
        assert exp is not None
        if isinstance(exp, (int, float)):
            return float(exp), ""
        exp = str(exp).replace(" ", "").lower()
        i = 1 if exp.startswith("-") else 0
        while i < len(exp):
            if (exp[i] < "0" or exp[i] > "9") and exp[i] != ".":
                break
            i += 1
        return float(exp[:i]), exp[i:]
    except (ValueError, AssertionError):
        raise ValueError(f"Invalid expression {exp}")
