from pytest import raises

from triad.utils.schema import (
    quote_name,
    safe_replace_out_of_quote,
    safe_split_and_unquote,
    safe_split_out_of_quote,
    split_quoted_string,
    unquote_name,
)


def test_quote_name():
    assert quote_name("") == "``"
    assert quote_name(" ") == "` `"
    assert quote_name("_") == "`_`"
    assert quote_name("`") == "````"
    assert quote_name("`a`") == "```a```"
    assert quote_name("`a`", quote="'") == "'`a`'"
    assert quote_name("大") == "`大`"
    assert quote_name("`a") == "```a`"

    assert quote_name("a") == "a"
    assert quote_name("Abc") == "Abc"


def test_unquote_name():
    raises(ValueError, lambda: unquote_name(""))
    assert unquote_name("``") == ""
    assert unquote_name("` `") == " "
    assert unquote_name("````") == "`"
    assert unquote_name("`ab") == "`ab"
    assert unquote_name("'大'", quote="'") == "大"
    assert unquote_name("a b") == "a b"

    assert unquote_name("ab") == "ab"


def test_split_qupted_string():
    def _assert(orig, expected):
        res = "".join(
            orig[x[1] : x[2]] if not x[0] else "[" + orig[x[1] : x[2]][1:-1] + "]"
            for x in split_quoted_string(orig, quote="'")
        )
        assert res == expected

    _assert("", "")
    _assert(" ", " ")
    _assert("a''b", "a[]b")
    _assert("a''", "a[]")
    _assert("''b", "[]b")
    _assert("''", "[]")
    _assert("''", "[]")
    _assert("' '", "[ ]")
    _assert("''''", "['']")
    _assert("'''' ''", "[''] []")
    _assert(
        "SELECT * FROM 'tb' WHERE 'tb'.'x''''x'=5",
        "SELECT * FROM [tb] WHERE [tb].[x''''x]=5",
    )
    raises(SyntaxError, lambda: _assert("'", ""))
    raises(SyntaxError, lambda: _assert("'''", ""))
    raises(SyntaxError, lambda: _assert("a 'b. ", ""))


def test_replace():
    assert "" == safe_replace_out_of_quote("", " ", "")
    assert "221`11`1" == safe_replace_out_of_quote("111`11`1", "11", "22")
    assert ".'  '" == safe_replace_out_of_quote(". '  '   ", " ", "", quote="'")
    assert ".''" == safe_replace_out_of_quote(". '  '   ", " ", "", quote="`")


def test_split():
    assert safe_split_out_of_quote("", " ") == [""]
    assert safe_split_out_of_quote(" ", " ") == ["", ""]
    assert safe_split_out_of_quote("  ", " ") == ["", "", ""]
    assert safe_split_out_of_quote("  ", " ", max_split=1) == ["", " "]
    assert safe_split_out_of_quote("  ,", ",") == ["  ", ""]

    assert safe_split_out_of_quote("',',", ",", quote="'") == ["','", ""]
    assert safe_split_out_of_quote("',',',,'", ",", quote="'") == ["','", "',,'"]


def test_split_and_unquote():
    assert safe_split_and_unquote(" a , ' , ' , b ", quote="'") == ["a", " , ", "b"]
    assert safe_split_and_unquote(" a , ' '' ' , b ", quote="'") == ["a", " ' ", "b"]
    # some tolerance of unquoted strings
    assert safe_split_and_unquote(" a , c d , b ", quote="'") == ["a", "c d", "b"]
    assert safe_split_and_unquote(
        " a , '', b, ", quote="'", on_unquoted_empty="keep"
    ) == ["a", "", "b", ""]
    assert safe_split_and_unquote(
        " a , '', b, ", quote="'", on_unquoted_empty="ignore"
    ) == ["a", "", "b"]
    with raises(ValueError):
        safe_split_and_unquote(" a , '', b, ", quote="'", on_unquoted_empty="throw")
