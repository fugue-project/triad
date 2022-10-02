from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from .assertion import assert_or_throw
from .string import validate_triad_var_name


def normalize_names(names: List[Any]) -> Dict[Any, str]:
    """Normalize dataframe column names to follow Fugue column naming
    rules. It only operates on names that are not valid to Fugue.

    It tries to minimize the changes to the original name. Special characters
    will be converted to ``_``, but if this does not provide a valid and
    unique column name, more transformation will be done.

    .. note::

        This is a temporary solution before :class:`~.triad.collections.schema.Schema`
        can take arbitrary names

    .. admonition:: Examples

        * ``[0,1]`` => ``{0:"_0", 1:"_1"}``
        * ``["1a","2b"]`` => ``{"1a":"_1a", "2b":"_2b"}``
        * ``["*a","-a"]`` => ``{"*a":"_a", "-a":"_a_1"}``

    :param names: the columns names of a dataframe
    :return: the rename operations as a dict, key is the original column
        name, value is the new valid name.
    """
    assert_or_throw(len(names) > 0, ValueError("names is empty"))
    assert_or_throw(
        len(set(names)) == len(names), ValueError(f"duplicated names found in {names}")
    )
    dup_ct = defaultdict(int)
    dup_ct[""] = 1
    result: Dict[Any, str] = {}
    _names: List[str] = []
    for _name in names:
        if isinstance(_name, str) and validate_triad_var_name(_name):
            dup_ct[_name] += 1
        else:
            _names.append(_name)
    for _name in _names:
        name = None if _name is None else str(_name)
        nn = _normalize_name(name)
        if dup_ct[nn] > 0:
            while dup_ct[nn] > 0:
                orig_nn = nn
                nn = nn + "_" + str(dup_ct[nn])
            dup_ct[orig_nn] += 1
        dup_ct[nn] += 1
        if not isinstance(_name, str) or _name != nn:
            result[_name] = nn
    return result


def _normalize_name(name: Optional[str]) -> str:
    if name is None:
        return ""
    if validate_triad_var_name(name):
        return name
    name = name.strip()
    if name == "":
        return ""
    name = "".join(_normalize_chars(name))
    if name[0].isdigit():
        name = "_" + name
    if validate_triad_var_name(name):
        return name
    return ""


def _normalize_chars(name: str) -> Iterable[str]:
    for c in name:
        i = ord(c)
        if i < len(_VALID_CHARS) and _VALID_CHARS[i]:
            yield c
        else:
            yield "_"


def _get_valid_signs():
    signs = [False] * 128
    for i in range(len(signs)):
        c = chr(i)
        if c.isalpha() or c.isdigit() or c == "_":
            signs[i] = True
    return signs


_VALID_CHARS = _get_valid_signs()
