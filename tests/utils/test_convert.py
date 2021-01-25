import builtins
import urllib  # must keep for testing purpose
import urllib.request  # must keep for testing purpose
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import tests.utils.convert_examples as ex
from pytest import raises
from tests.utils.convert_examples import BaseClass, Class2
from tests.utils.convert_examples import SubClass
from tests.utils.convert_examples import SubClass as SubClassSame
from triad.utils.convert import (
    _parse_value_and_unit,
    as_type,
    get_caller_global_local_vars,
    get_full_type_path,
    str_to_instance,
    str_to_object,
    str_to_type,
    to_bool,
    to_datetime,
    to_function,
    to_instance,
    to_size,
    to_timedelta,
    to_type,
)

_GLOBAL_DUMMY = 1


def test_to_size():
    raises(ValueError, lambda: to_size(None))
    raises(ValueError, lambda: to_size(""))
    raises(ValueError, lambda: to_size("abc"))
    raises(AssertionError, lambda: to_size("-1"))
    raises(AssertionError, lambda: to_size(-1))
    raises(ValueError, lambda: to_size("1xx"))
    assert 0 == to_size(0)
    assert 1 == to_size(1)
    assert 1 == to_size(1.9)
    assert 10 == to_size(" 1 0 B ")
    assert 10 * 1024 == to_size(" 10k")
    assert 10 * 1024 * 1024 == to_size(" 10 m b")
    assert 10 * 1024 * 1024 * 1024 == to_size("10g")
    assert 10 * 1024 * 1024 * 1024 * 1024 == to_size("10tb")
    assert int(1.1 * 1024 * 1024) == to_size(" 1 . 1 mb ")


def test_parse_value_and_unit():
    raises(ValueError, lambda: _parse_value_and_unit(None))
    raises(ValueError, lambda: _parse_value_and_unit(""))
    raises(ValueError, lambda: _parse_value_and_unit("abc"))
    assert (1.0, "") == _parse_value_and_unit(1)
    assert (1.1, "") == _parse_value_and_unit(1.1)
    assert (1.1, "") == _parse_value_and_unit(1.1)
    assert (1.1, "") == _parse_value_and_unit(np.float32(1.1))
    assert (1.0, "") == _parse_value_and_unit(" 1 ")
    assert (-1.0, "") == _parse_value_and_unit(" -1.0 ")
    assert (-1.0, "m10") == _parse_value_and_unit(" - 1 . 0 m 1 0 ")


def test_str_to_object():
    class _Mock(object):
        def __init__(self, x=1):
            self.x = x

        def test(self):
            assert self.x == str_to_object("self.x")

    m = _Mock()
    m.test()
    assert BaseClass == str_to_object("tests.utils.BaseClass")
    assert BaseClass == str_to_object("tests.utils.convert_examples.BaseClass")
    assert SubClass == str_to_object("SubClass")
    assert SubClassSame == str_to_object("SubClassSame")
    assert RuntimeError == str_to_object("RuntimeError")
    assert _Mock == str_to_object("_Mock")
    assert 1 == str_to_object("m.x")
    assert 1 == str_to_object("m2.x", local_vars={"m2": m})
    raises(ValueError, lambda: str_to_object(""))
    raises(ValueError, lambda: str_to_object("xxxx"))
    raises(ValueError, lambda: str_to_object("xx.xx"))


def test_str_to_type():
    assert BaseClass == str_to_type("tests.utils.BaseClass")
    assert BaseClass == str_to_type("tests.utils.convert_examples.BaseClass")
    assert SubClass == str_to_type("SubClass", BaseClass)
    assert SubClassSame == str_to_type("SubClassSame")
    raises(
        TypeError, lambda: str_to_type("tests.utils.convert_examples.BaseClass", int)
    )
    raises(TypeError, lambda: str_to_type("tests.utils.convert_examples.BaseClass1"))
    raises(TypeError, lambda: str_to_type("pytest.raises"))
    raises(TypeError, lambda: str_to_type("__dummy_impossible__"))

    assert RuntimeError == str_to_type("RuntimeError")
    assert RuntimeError == str_to_type("RuntimeError", Exception)
    raises(TypeError, lambda: str_to_type("RuntimeError", int))

    # test a full type path that only root was imported
    str_to_type("urllib.request.OpenerDirector")

    # test a full type path that was never imported
    str_to_type("shutil.Error")
    str_to_type("http.HTTPStatus")

    # class and subclass
    class T(object):
        def __init__(self):
            self.x = 10

        class _TS(object):
            pass

    assert T == str_to_type("T")
    assert T._TS == str_to_type("T._TS")


def test_str_to_instance():
    i = str_to_instance("tests.utils.Class2")
    assert isinstance(i, Class2)
    assert i.s == "000"
    i = str_to_instance("tests.utils.Class2", args=[1, 2], kwargs=dict(c=3))
    assert isinstance(i, Class2)
    assert i.s == "123"
    i = str_to_instance("tests.utils.Class2", args=[1, 2], kwargs=dict(c=3))
    assert isinstance(i, Class2)
    assert i.s == "123"


def test_obj_to_type():
    assert to_type(None) is type(None)
    raises(TypeError, lambda: to_type(None, str))
    assert issubclass(to_type(123), int)
    raises(TypeError, lambda: to_type(123, str))
    i = to_type("tests.utils.Class2")
    assert issubclass(i, Class2)
    i = to_type(Class2)
    assert issubclass(i, Class2)
    raises(TypeError, lambda: to_type(Class2, BaseClass))
    raises(TypeError, lambda: to_type("tests.utils.Class2", BaseClass))

    assert ex.__Dummy__ is not __Dummy__
    assert to_type("tests.utils.convert_examples.__Dummy__") is ex.__Dummy__


def test_obj_to_instance():
    raises(ValueError, lambda: to_instance(None))
    i = to_instance("tests.utils.Class2", args=[1, 2, 3])
    assert isinstance(i, Class2)
    assert i.s == "123"
    i = to_instance(Class2, kwargs={"a": 1, "b": 2, "c": 3})
    assert isinstance(i, Class2)
    assert i.s == "123"
    i = to_instance(Class2)
    assert isinstance(i, Class2)
    i = to_instance(Class2, args=[1, 2, 3])
    assert isinstance(i, Class2)
    raises(TypeError, lambda: to_instance(Class2, BaseClass, args=[1, 2, 3]))
    i = to_instance(Class2("a:int"))
    assert isinstance(i, Class2)
    raises(TypeError, lambda: to_instance(Class2(), BaseClass))
    raises(ValueError, lambda: to_instance(Class2(), args=[1]))
    raises(ValueError, lambda: to_instance(Class2(), kwargs={"a": 1}))

    assert ex.__Dummy__ is not __Dummy__
    assert type(to_instance("__Dummy__")) is __Dummy__
    assert type(to_instance("tests.utils.convert_examples.__Dummy__")) is ex.__Dummy__
    assert type(to_instance("tests.utils.convert_examples.__Dummy__")) is ex.__Dummy__


def test_obj_to_function():
    def _mock():
        pass

    assert _mock == to_function("_mock")

    f = to_function("dummy_for_test")
    assert f == dummy_for_test
    f = to_function(dummy_for_test)
    assert f == dummy_for_test
    f = to_function("open")
    assert f == open
    f = to_function("tests.utils.test_convert.dummy_for_test")
    assert f == dummy_for_test
    f = to_function("triad.utils.convert.to_instance")
    assert f == to_instance
    raises(AttributeError, lambda: to_function(None))
    raises(AttributeError, lambda: to_function("asdfasdf"))
    raises(AttributeError, lambda: to_function("BaseClass"))

    assert to_function("min") == builtins.min

    class _Mock(object):
        def x(self, p=10):
            return p * 10

        @property
        def xx(self):
            return 0

    m = _Mock()
    assert to_function("m.x") == m.x
    assert 30 == to_function("m.x")(3)
    raises(AttributeError, lambda: to_function("m.xx"))


def test_to_bool():
    raises(TypeError, lambda: to_bool(None))
    assert to_bool("TRUE")
    assert to_bool(True)
    assert to_bool("1")
    assert to_bool(1)
    assert to_bool("Yes")
    assert not to_bool("FALSE")
    assert not to_bool(False)
    assert not to_bool("0")
    assert not to_bool(0)
    raises(TypeError, lambda: to_bool("x"))


def test_to_datetime():
    raises(TypeError, lambda: to_datetime(None))
    dt = datetime.now()
    assert dt == to_datetime(dt)
    assert dt == to_datetime(str(dt))
    assert datetime(2019, 5, 18) == to_datetime("2019-05-18")
    assert datetime(2019, 5, 18, 10, 11, 12) == to_datetime("2019-05-18 10:11:12")
    assert datetime(2019, 5, 18) == to_datetime(date(2019, 5, 18))
    raises(TypeError, lambda: to_datetime("x"))
    raises(TypeError, lambda: to_datetime(123))


def test_to_timedelta():
    raises(TypeError, lambda: to_timedelta(None))
    dt = timedelta(days=2)
    assert dt == to_timedelta(dt)
    assert dt == to_timedelta("2d")
    assert dt == to_timedelta("48h")
    dt = timedelta(days=2, minutes=1)
    assert dt == to_timedelta("2 day 1 min")
    assert dt == to_timedelta("2d1m")
    dt = pd.Timedelta("2d")
    assert timedelta(days=2) == to_timedelta(dt)
    raises(TypeError, lambda: to_timedelta("x"))
    assert timedelta() == to_timedelta(0)
    assert timedelta(seconds=2.5) == to_timedelta(2.5)
    assert timedelta(seconds=2.1) == to_timedelta(np.float64(2.1))
    assert timedelta(seconds=2) == to_timedelta(np.int32(2))
    assert timedelta.max == to_timedelta("Max")
    assert timedelta.max == to_timedelta("InF")
    assert timedelta.min == to_timedelta("mIn")
    assert timedelta.min == to_timedelta("-InF")


def test_as_type():
    assert 10 == as_type(10, int)
    assert 10 == as_type("10", int)
    assert 1 == as_type(1.1, int)
    assert "10" == as_type(10, str)
    assert not as_type(False, bool)
    assert not as_type("no", bool)
    assert as_type("a:int", Class2).s == Class2("a:int").s
    assert as_type(None, Class2).s == Class2(None).s
    assert timedelta(days=2) == as_type("2d", timedelta)
    assert datetime(2019, 5, 18) == as_type("2019-05-18", datetime)


def test_get_full_type_path():
    assert "tests.utils.test_convert.dummy_for_test" == get_full_type_path(
        dummy_for_test
    )
    raises(TypeError, lambda: get_full_type_path(lambda x: x + 1))
    assert "tests.utils.test_convert.__Dummy__" == get_full_type_path(__Dummy__)
    assert "tests.utils.convert_examples.SubClass" == get_full_type_path(SubClassSame)
    raises(TypeError, lambda: get_full_type_path(None))
    assert "builtins.int" == get_full_type_path(int)
    assert "builtins.dict" == get_full_type_path(dict)
    assert "builtins.Exception" == get_full_type_path(Exception)

    assert "builtins.int" == get_full_type_path(123)
    assert "builtins.str" == get_full_type_path("ad")
    assert "tests.utils.test_convert.__Dummy__" == get_full_type_path(__Dummy__())


def test_get_caller_global_local_vars():
    def f1():
        f1__var = 1
        f2__var = 2
        f3__var = 3

        def f2():
            f2__var = 22

            def f3():
                f3__var = 33

                def f4():
                    f4__var = 44
                    g, l = get_caller_global_local_vars(None, None, start=0, end=0)
                    assert "f4__var" in l
                    assert 44 == l["f4__var"]
                    assert "f3__var" not in l
                    assert 1 == g["_GLOBAL_DUMMY"]

                    g, l = get_caller_global_local_vars(None, None, start=0, end=-1)
                    assert "f4__var" in l
                    assert "f3__var" in l
                    assert 33 == l["f3__var"]

                    g, l = get_caller_global_local_vars(None, None, start=-2, end=-2)
                    assert "f4__var" not in l
                    assert "f3__var" not in l
                    assert 22 == l["f2__var"]

                    g, l = get_caller_global_local_vars(None, None, start=-2, end=-3)
                    assert "f4__var" not in l
                    assert 22 == l["f2__var"]
                    assert 1 == l["f1__var"]
                    assert 3 == l["f3__var"]

                    g, l = get_caller_global_local_vars(None, None, start=0, end=-100)
                    assert 22 == l["f2__var"]
                    assert 1 == l["f1__var"]
                    assert 33 == l["f3__var"]
                    assert 44 == l["f4__var"]
                    assert 1 == g["_GLOBAL_DUMMY"]

                    g, l = get_caller_global_local_vars(
                        {"k": 1}, None, start=0, end=-100
                    )
                    assert 22 == l["f2__var"]
                    assert 1 == l["f1__var"]
                    assert 33 == l["f3__var"]
                    assert 44 == l["f4__var"]
                    assert 1 == g["k"]
                    assert "_GLOBAL_DUMMY" not in g

                f4()

            f3()

        f2()

    f1()


# This is for test_obj_to_function
def dummy_for_test():
    pass


# This is to test *_to_type with first=False
class __Dummy__(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
