from tests.utils.dispatcher_examples import dtest
from triad import run_at_def


@dtest.candidate(lambda x: isinstance(x, int))
def dtest1(a: int) -> int:
    return a


@dtest.candidate(lambda x: isinstance(x, str))
def dtest2(a: str) -> int:
    return len(a)


def dtest3(a: dict) -> int:
    return len(a)


@run_at_def
def register():
    dtest.register(dtest3, lambda x: isinstance(x, dict))
