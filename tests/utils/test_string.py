from triad.utils.string import is_quoteless_column_name


def test_is_quoteless_column_name():
    assert not is_quoteless_column_name("")
    assert not is_quoteless_column_name("`")
    assert not is_quoteless_column_name(" ")
    assert not is_quoteless_column_name("中国")
    assert not is_quoteless_column_name("_中国")
    assert not is_quoteless_column_name("مثال")

    assert is_quoteless_column_name("a")
    assert is_quoteless_column_name("abc")
    assert is_quoteless_column_name("_")
    assert is_quoteless_column_name("__")
