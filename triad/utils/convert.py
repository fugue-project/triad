import datetime
import importlib
import inspect
from pydoc import locate
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import six
from ciso8601 import parse_datetime

EMPTY_ARGS: List[Any] = []
EMPTY_KWARGS: Dict[str, Any] = {}


def str_to_type(
    s: str, expected_base_type: Optional[type] = None, first: bool = False
) -> type:
    """Given a string expression, find the first/last type from all import libraries.
    If the expression contains `.`, it's supposed to be a full path of the type
    including modules, and this function will use `pydoc.locate` to find the type.
    Otherwise, it will firstly check if it is a built in type, then search
    on the call stack.

    :param s: type expression, for example `triad.utils.iter.Slicer` or `str`
    :param expected_base_type: base class type that must satisfy, defaults to None
    :param first: to return the nearest (True) or farthest (False) matching one

    :raises TypeError: unable to find a matching type

    :return: found type
    """
    t = list(str_to_types(s, expected_base_type))
    if len(t) == 0:
        if expected_base_type is not None:
            raise TypeError(f"{s} is not type {str(expected_base_type)}")
        else:
            raise TypeError(f"{s} is not type")
    if first:
        return t[0]
    return t[-1]


def str_to_types(s: str, expected_base_type: Optional[type] = None) -> Iterable[type]:
    """Given a string expression, find the type from all import libraries.
    If the expression contains `.`, it's supposed to be a full path of the type
    including modules, and this function will use `pydoc.locate` to find the type.
    Otherwise, it will firstly check if it is a built in type, then search
    on the call stack, starting from nearest level.

    :param s: type expression, for example `triad.utils.iter.Slicer` or `str`
    :param expected_base_type: base class type that must satisfy, defaults to None

    :yield: found types, starting from the top on call stack
    """
    if "." in s:
        t = locate(s)
        if isinstance(t, type) and (
            expected_base_type is None or issubclass(t, expected_base_type)
        ):
            yield t
    else:
        for tt in str_to_types("builtins." + s, expected_base_type):
            yield tt
        for frm in inspect.stack():
            t = frm[0].f_globals["__name__"] + "." + s
            for tt in str_to_types(str(t), expected_base_type):
                yield tt


def str_to_instance(
    s: str,
    expected_base_type: Optional[type] = None,
    first: bool = False,
    args: List[Any] = EMPTY_ARGS,
    kwargs: Dict[str, Any] = EMPTY_KWARGS,
) -> Any:
    """Use :func:'~triad.utils.convert.str_to_type' to find a matching type
    and instantiate

    :param s: see :func:'~triad.utils.convert.str_to_type'
    :param expected_base_type: see :func:'~triad.utils.convert.str_to_type'
    :param first: see :func:'~triad.utils.convert.str_to_type'
    :param args: args to instantiate the type
    :param kwargs: kwargs to instantiate the type

    :return: the instantiated the object
    """
    t = str_to_type(s, expected_base_type, first)
    return t(*args, **kwargs)


def to_type(
    s: Any, expected_base_type: Optional[type] = None, first: bool = False
) -> type:
    """Convert an object `s` to `type`
    * if `s` is `str`: see :func:'~triad.utils.convert.str_to_type'
    * if `s` is `type`: check `expected_base_type` and return itself
    * else: check `expected_base_type` and return itself

    :param s: see :func:'~triad.utils.convert.str_to_type'
    :param expected_base_type: see :func:'~triad.utils.convert.str_to_type'
    :param first: see :func:'~triad.utils.convert.str_to_type'

    :raises TypeError: if no matching type found

    :return: the matching type
    """
    if isinstance(s, str):
        return str_to_type(s, expected_base_type, first)
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
    first: bool = False,
    args: List[Any] = EMPTY_ARGS,
    kwargs: Dict[str, Any] = EMPTY_KWARGS,
) -> Any:
    """If s is str or type, then use :func:'~triad.utils.convert.to_type' to find
    matching type and instantiate. Otherwise return s if it matches constraints

    :param s: see :func:'~triad.utils.convert.to_type'
    :param expected_base_type: see :func:'~triad.utils.convert.to_type'
    :param first: see :func:'~triad.utils.convert.to_type'
    :param args: args to instantiate the type
    :param kwargs: kwargs to instantiate the type

    :raises ValueError: if s is an instance but not a (sub)type of `expected_base_type`
    :raises TypeError: if s is an instance, args and kwargs must be empty

    :return: the instantiated object
    """
    if s is None:
        raise ValueError("None can't be converted to instance")
    if isinstance(s, (str, type)):
        t = to_type(s, expected_base_type, first)
        return t(*args, **kwargs)
    else:
        if expected_base_type is not None and not isinstance(s, expected_base_type):
            raise TypeError(f"{str(s)} is not a subclass of {str(expected_base_type)}")
        if len(args) > 0 or len(kwargs) > 0:
            raise ValueError(f"Can't instantiate {str(s)} with different parameters")
        return s


def to_function(s: Any, first: bool = False) -> Any:  # noqa: C901
    """For an expression, it will try to find the nearest or farthest
    correspondent function in all imported libraries.
    Otherwise if it is a callable and not a class, it will return the input itself.

    :params s: a string expression or a callable
    :param first: to return the nearest (True) or farthest (False) matching one

    :raises AttributeError: if unable to find such a function

    :return: the nearest (first=True) or farthest (False) matching function
    """
    funcs = list(to_functions(s))
    if len(funcs) == 0:
        raise AttributeError(f"{s} is not a valid function")
    if first:
        return funcs[0]
    return funcs[-1]


def to_functions(s: Any) -> Iterable[Any]:  # noqa: C901
    """For an expression, it will try to find all correspondent function
    in all imported libraries, starting from the nearest on call stack.
    Otherwise if it is a callable and not a class, it will return the input itself.

    :params s: a string expression or a callable

    :raises AttributeError: if unable to find such a function

    :yield: matching functions
    """
    if s is None:
        raise AttributeError("None can't be converted to function")
    func: Any = None
    if isinstance(s, str):
        if "." in s:
            mod_name, func_name = s.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            try:
                func = getattr(mod, func_name)
            except AttributeError:
                func = None
        else:
            for f in to_functions("builtins." + s):
                yield f
            for frm in inspect.stack():
                t = frm[0].f_globals["__name__"] + "." + s
                for f in to_functions(str(t)):
                    yield f
    else:
        func = s
    if callable(func) and not isinstance(func, six.class_types):
        yield func


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
