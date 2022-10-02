from triad.utils.rename import normalize_names, _normalize_name
from pytest import raises


def test_normalize_names():
    raises(ValueError, lambda: normalize_names([]))
    raises(ValueError, lambda: normalize_names([0, 0]))
    raises(ValueError, lambda: normalize_names(["a", 0, "a", 1]))
    raises(ValueError, lambda: normalize_names(["大", "大"]))

    def _assert(input, output):
        res = normalize_names(input)
        assert res == output

    _assert(["a", "b", "c"], {})
    _assert(["a", "+b", "c"], {"+b": "_b"})
    _assert(["a", "+b", "_b"], {"+b": "_b_1"})
    _assert(["a", "+b", "-b", "_b"], {"+b": "_b_1", "-b": "_b_2"})

    _assert([None], {None: "_1"})
    _assert([None, ""], {None: "_1", "": "_2"})
    _assert([None, "_1"], {None: "_1_1"})
    _assert([None, "_1_1", "_1", ""], {None: "_1_1_1", "": "_1_1_2"})

    _assert([0, 2, 1], {0: "_0", 2: "_2", 1: "_1"})

    _assert(["大", "大大"], {"大": "_1", "大大": "_2"})
    _assert(["大a", "大b"], {"大a": "_a", "大b": "_b"})


def test_normalize_name():
    assert _normalize_name("a") == "a"

    assert _normalize_name(None) == ""
    assert _normalize_name("") == ""
    assert _normalize_name("  ") == ""
    assert _normalize_name(" a ") == "a"

    assert _normalize_name("1") == "_1"
    assert _normalize_name("1a") == "_1a"
    assert _normalize_name("1-a") == "_1_a"

    assert _normalize_name("_") == ""

    assert _normalize_name("$%^&") == ""  # no valid chars
    assert _normalize_name("$%(a^b&") == "___a_b_"
    assert _normalize_name("大大a大") == "__a_"
