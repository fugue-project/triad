import pandas as pd
from pytest import raises
from triad.collections.dataframe import PandasDataFrame
from triad.collections.schema import Schema
from triad.exceptions import InvalidOperationError, NoneArgumentError


def test_init():
    df = PandasDataFrame(schema="a:str,b:int")
    assert df.count() == 0
    assert df.schema == "a:str,b:int"
    assert Schema(df.native) == "a:str,b:int"

    pdf = pd.DataFrame([["a", 1], ["b", 2]])
    raises(NoneArgumentError, lambda: PandasDataFrame(pdf))
    df = PandasDataFrame(pdf, "a:str,b:str")
    assert [["a", "1"], ["b", "2"]] == df.native.values.tolist()
    df = PandasDataFrame(pdf, "a:str,b:int")
    assert [["a", 1], ["b", 2]] == df.native.values.tolist()
    df = PandasDataFrame(pdf, "a:str,b:double")
    assert [["a", 1.0], ["b", 2.0]] == df.native.values.tolist()

    pdf = pd.DataFrame([["a", 1], ["b", 2]], columns=["a", "b"])["b"]
    assert isinstance(pdf, pd.Series)
    df = PandasDataFrame(pdf, "b:str")
    assert [["1"], ["2"]] == df.native.values.tolist()
    df = PandasDataFrame(pdf, "b:double")
    assert [[1.0], [2.0]] == df.native.values.tolist()

    pdf = pd.DataFrame([["a", 1], ["b", 2]], columns=["x", "y"])
    df = PandasDataFrame(pdf)
    assert df.schema == "x:str,y:long"
    df = PandasDataFrame(pdf, "y:str,x:str")
    assert [["1", "a"], ["2", "b"]] == df.native.values.tolist()
    ddf = PandasDataFrame(df)
    assert [["1", "a"], ["2", "b"]] == ddf.native.values.tolist()
    assert df.native is ddf.native  # no real copy happened

    df = PandasDataFrame([["a", 1], ["b", "2"]], "x:str,y:double")
    assert [["a", 1.0], ["b", 2.0]] == df.native.values.tolist()

    df = PandasDataFrame([], "x:str,y:double")
    assert [] == df.native.values.tolist()

    raises(ValueError, lambda: PandasDataFrame(123))


def test_simple_methods():
    df = PandasDataFrame([], "a:str,b:int")
    assert df.as_pandas() is df.native
    assert df.empty()
    assert 0 == df.count()
    raises(IndexError, lambda: df.peek_array())
    raises(IndexError, lambda: df.peek_dict())
    assert df.is_local()

    df = PandasDataFrame([["a", 1], ["b", "2"]], "x:str,y:double")
    assert df.as_pandas() is df.native
    assert not df.empty()
    assert 2 == df.count()
    assert ["a", 1.0] == df.peek_array()
    assert dict(x="a", y=1.0) == df.peek_dict()


def test_drop():
    df = PandasDataFrame([], "a:str,b:int").drop(["a"])
    assert df.empty()
    assert df.schema == "b:int"
    raises(InvalidOperationError, lambda: df.drop(["b"]))  # can't be empty
    raises(InvalidOperationError, lambda: df.drop(["x"]))  # cols must exist

    df = PandasDataFrame([["a", 1]], "a:str,b:int").drop(["a"])
    assert df.schema == "b:int"
    raises(InvalidOperationError, lambda: df.drop(["b"]))  # can't be empty
    raises(InvalidOperationError, lambda: df.drop(["x"]))  # cols must exist
    assert [[1]] == df.as_pandas().values.tolist()


def test_as_array():
    df = PandasDataFrame([], "a:str,b:int")
    assert [] == df.as_array()
    assert [] == df.as_array(type_safe=True)
    assert [] == list(df.as_array_iterable())
    assert [] == list(df.as_array_iterable(type_safe=True))

    df = PandasDataFrame([["a", 1]], "a:str,b:int")
    assert [["a", 1]] == df.as_array()
    assert [["a", 1]] == df.as_array(["a", "b"])
    assert [[1, "a"]] == df.as_array(["b", "a"])

    # prevent pandas auto type casting
    df = PandasDataFrame([[1.0, 1.1]], "a:double,b:int")
    assert [[1.0, 1]] == df.as_array()
    assert isinstance(df.as_array()[0][0], float)
    assert isinstance(df.as_array()[0][1], int)
    assert [[1.0, 1]] == df.as_array(["a", "b"])
    assert [[1, 1.0]] == df.as_array(["b", "a"])
