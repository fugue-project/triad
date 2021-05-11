import pickle

from pytest import raises
from triad.exceptions import NoneArgumentError
from triad.utils.assertion import assert_arg_not_none, assert_or_throw


def test_assert_or_throw():
    assert_or_throw(True)
    raises(AssertionError, lambda: assert_or_throw(False))
    raises(AssertionError, lambda: assert_or_throw(False, "test"))
    raises(AssertionError, lambda: assert_or_throw(False, 123))
    raises(TypeError, lambda: assert_or_throw(False, TypeError()))
    
    # lazy evaluation
    raises(TypeError, lambda: assert_or_throw(False, lambda: TypeError()))

    def fail_without_lazy():
        raise TypeError

    assert_or_throw(True, fail_without_lazy)
    raises(TypeError, lambda: assert_or_throw(False, fail_without_lazy))

    # serialization + lazy
    mock = pickle.loads(pickle.dumps(Mock()))
    raises(TypeError, lambda: mock.t1(False))
    mock.t1(True)

    raises(NotImplementedError, lambda: mock.t2(False))
    mock.t2(True)


def test_assert_arg_not_none():
    assert_arg_not_none(1)
    with raises(NoneArgumentError) as err:
        assert_arg_not_none(None, "a")
    assert "a can't be None" == err.value.args[0]
    with raises(NoneArgumentError) as err:
        assert_arg_not_none(None, "a", "b")
    assert "a can't be None" == err.value.args[0]
    with raises(NoneArgumentError) as err:
        assert_arg_not_none(None, "", msg="b")
    assert "b" == err.value.args[0]
    with raises(NoneArgumentError) as err:
        assert_arg_not_none(None, None, msg="b")
    assert "b" == err.value.args[0]


class Mock:
    def t1(self, v):
        def fail_without_lazy():
            raise TypeError

        assert_or_throw(v, fail_without_lazy)

    def t2(self, v):
        assert_or_throw(v, lambda: NotImplementedError())
