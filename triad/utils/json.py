import json
from typing import Any, Dict, Hashable, List, Tuple


def loads_no_dup(json_str: str) -> Any:
    """Load json string, and raise KeyError if there are duplicated keys

    :param json_str: json string
    :raises KeyError: if there are duplicated keys
    :return: the parsed object
    """
    return json.loads(json_str, object_pairs_hook=check_for_duplicate_keys)


def check_for_duplicate_keys(
    ordered_pairs: List[Tuple[Hashable, Any]]
) -> Dict[Any, Any]:
    """Raise ValueError if a duplicate key exists in provided ordered list of pairs,
    otherwise return a dict.

    Example:
    >>> json.loads('{"x": 1, "x": 2}', object_pairs_hook=check_for_duplicate_keys)

    :raises KeyError: if there is duplicated key
    """
    dict_out: Dict[Any, Any] = {}
    for key, val in ordered_pairs:
        if key in dict_out:
            raise KeyError(f"Duplicate key: {key}")
        else:
            dict_out[key] = val
    return dict_out
