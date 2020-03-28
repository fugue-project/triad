from typing import Tuple, Any


def to_size(exp: Any) -> int:
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
