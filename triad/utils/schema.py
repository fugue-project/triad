from typing import Iterable, List, Tuple

from triad.constants import TRIAD_VAR_QUOTE

from .assertion import assert_or_throw
from .string import validate_triad_var_name


def unquote_name(name: str, quote: str = TRIAD_VAR_QUOTE) -> str:
    """If the input is quoted, then get the inner string,
    otherwise do nothing.

    :param name: the name string
    :param quote: the quote char, defaults to `
    :return: the value without `
    """
    if validate_triad_var_name(name):
        return name
    if len(name) >= 2 and name[0] == name[-1] == quote:
        return name[1:-1].replace(quote + quote, quote)
    name = name.strip()
    assert_or_throw(
        len(name) > 0,
        ValueError("empty string is invalid"),
    )
    return name


def quote_name(name: str, quote: str = TRIAD_VAR_QUOTE) -> str:
    """Add quote ` for strings that are not a valid triad var name.

    :param name: the name string
    :param quote: the quote char, defaults to `
    :return: the quoted(if necessary) string
    """
    if validate_triad_var_name(name):
        return name
    return quote + name.replace(quote, quote + quote) + quote


def move_to_unquoted(expr: str, p: int, quote=TRIAD_VAR_QUOTE) -> int:
    """When ``p`` is on a quote, find the position next to the end of
    the quoted part

    :param expr: the original string
    :param p: the current position of ``expr``, and it should be a quote
    :param quote: the quote character
    :raises SyntaxError: if there is an open quote detected
    :return: the position next to the end of the quoted part
    """
    e = p + 1
    le = len(expr)
    while e < le:
        if expr[e] == quote:
            if e + 1 < le and expr[e + 1] == quote:  # escape (``)
                e = e + 2
            else:
                return e + 1
        else:
            e += 1
    raise SyntaxError(f"{expr} contains open quote {quote}")


def split_quoted_string(
    s: str, quote=TRIAD_VAR_QUOTE
) -> Iterable[Tuple[bool, int, int]]:
    """Split ``s`` to a sequence of quoted and unquoted parts.

    :param s: the original string
    :param quote: the quote character

    :yield: the tuple in the format of ``is_quoted, start, end``
    """
    b, e = 0, 0
    le = len(s)
    while e < le:
        if s[e] == quote:
            if e > b:
                yield False, b, e
            b = e
            e = move_to_unquoted(s, e, quote=quote)
            yield True, b, e
            b = e
        else:
            e += 1
    if b < le:
        yield False, b, le


def safe_search_out_of_quote(
    s: str, chars: str, quote=TRIAD_VAR_QUOTE
) -> Iterable[Tuple[int, str]]:
    """Search for chars out of the quoted parts

    :param s: the original string
    :param chars: the charaters to find
    :param quote: the quote character
    :yield: the tuple in format of ``position, char``
    """
    for rg in split_quoted_string(s, quote=quote):
        if not rg[0]:
            for i in range(rg[1], rg[2]):
                if s[i] in chars:
                    yield i, s[i]


def safe_replace_out_of_quote(
    s: str, find: str, replace: str, quote=TRIAD_VAR_QUOTE
) -> str:
    """Replace strings out of the quoted part

    :param s: the original string
    :param find: the string to find
    :param replace: the string used to replace
    :param quote: the quote character

    :return: the string with the replacements
    """
    return "".join(
        s[rg[1] : rg[2]] if rg[0] else s[rg[1] : rg[2]].replace(find, replace)
        for rg in split_quoted_string(s, quote=quote)
    )


def safe_split_out_of_quote(
    s: str,
    sep_chars: str,
    max_split: int = -1,
    quote: str = TRIAD_VAR_QUOTE,
) -> List[str]:
    b = 0
    if max_split == 0 or len(s) == 0:
        return [s]
    res: List[str] = []
    for p, _ in safe_search_out_of_quote(s, sep_chars, quote=quote):
        res.append(s[b:p])
        b = p + 1
        if len(res) == max_split:
            break
    res.append(s[b:])
    return res


def safe_split_and_unquote(
    s: str,
    sep_char: str = ",",
    quote: str = TRIAD_VAR_QUOTE,
    on_unquoted_empty: str = "keep",
) -> List[str]:
    """Split the string and unquote every part

    .. admonition:: Examples

        ``" a , ` b ` , c "`` => ``["a", " b ","c"]``

    :param s: the original string
    :param sep_char: the split character, defaults to ","
    :param quote: the quote character
    :param on_unquoted_empty: can be ``keep``, ``ignore`` or
        ``throw``, defaults to "keep"
    :raises ValueError: if there are empty but unquoted parts and
        ``on_unquoted_empty`` is ``throw``
    :return: the unquoted parts.
    """
    res: List[str] = []
    for _s in safe_split_out_of_quote(s, sep_char, quote=quote):
        s = _s.strip()
        if s == "":
            if on_unquoted_empty == "keep":
                res.append(s)
            elif on_unquoted_empty == "ignore":
                continue
            else:
                raise ValueError(f"empty is not allowed in {s}")
        else:
            res.append(unquote_name(s, quote=quote))
    return res
