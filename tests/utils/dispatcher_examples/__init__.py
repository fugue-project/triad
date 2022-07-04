from triad import conditional_dispatcher
from typing import Any


@conditional_dispatcher(entry_point="tests.plugins")
def dtest(a: Any) -> int:
    raise NotImplementedError
