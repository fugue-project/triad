import inspect
from typing import Any, Dict, List, Optional, Tuple
from triad.utils.convert import to_type, to_function


def extract_function_io_types(
    func: Any, ignore_unknown_type: bool = False
) -> Tuple[List[Tuple[str, Optional[type]]], Optional[type]]:
    """Inspect the function and return input output specs

    :param func: input function

    :return: input name, type tuples and output type (None if no return)
    """
    sig = inspect.signature(func)
    inputs: List[Tuple[str, Optional[type]]] = []
    for k, v in sig.parameters.items():
        ktype: Optional[type] = None
        try:
            t = to_type(v.annotation)
            if t != inspect._empty:  # type: ignore
                ktype = t
        except Exception:
            if not ignore_unknown_type:
                raise
        inputs.append((k, ktype))
    otype: Optional[type] = None
    try:
        t = to_type(sig.return_annotation)
        if t != inspect._empty and t != type(None):  # type: ignore # noqa E721
            otype = t
    except Exception:
        if not ignore_unknown_type:
            raise
    return inputs, otype


def extract_function_arg_names(func: Any) -> List[str]:
    """Extract function argument names

    :param func: the input function
    :return: a list of names
    """
    sig = inspect.signature(func)
    args: List[str] = []
    for k, v in sig.parameters.items():
        raw = v.__str__()
        name = k
        if raw.startswith("**"):
            name = "**" + k
        elif raw.startswith("*"):
            name = "*" + k
        args.append(name)
    return args


def safe_invoke(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Safely invoke a function when the given arguments is a superset
    of what can be accepted

    :param func: the function
    :raises TypeError: the arguments don't match and can't be tolerated
    """
    f = to_function(func)
    sig = inspect.signature(f)
    ag: List[Any] = list(args)
    kag: Dict[str, Any] = dict(kwargs)
    a: List[Any] = []
    for _, v in sig.parameters.items():
        k = v.name
        if not v.__str__().startswith("*"):
            if k in kag:
                a.append(kag.pop(k))
            elif len(ag) > 0:
                a.append(ag.pop(0))
            elif v.default is not inspect.Parameter.empty:
                a.append(v.default)
            else:
                raise TypeError(f"Unable to set value for {k}")
        elif v.__str__().startswith("**"):
            return f(*a, *ag, **kag)
        else:
            a += ag
            ag = []
    return f(*a)
