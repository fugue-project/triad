from pytest import raises
from triad.utils.function import (extract_function_arg_names,
                                  extract_function_io_types, safe_invoke)


def test__extract_func_id():
    assert ([], None) == extract_function_io_types(efi0)
    assert ([("a", int), ("b", None)], None) == extract_function_io_types(efi1)
    assert ([("a", None)], None) == extract_function_io_types(efi2)
    assert ([("a", None), ("b", int), ("arg", None), ("karg", None)],
            int) == extract_function_io_types(efi3)
    assert ([("a", int), ("b", None)], None) == extract_function_io_types(efi4, True)
    raises(TypeError, lambda: extract_function_io_types(efi4))
    raises(TypeError, lambda: extract_function_io_types(efi4, False))
    assert ([("a", int)], None) == extract_function_io_types(efi5, True)
    raises(TypeError, lambda: extract_function_io_types(efi5))
    raises(TypeError, lambda: extract_function_io_types(efi5, False))


def test__extract_function_arg_names():
    assert [] == extract_function_arg_names(si_0)
    assert ["a", "b"] == extract_function_arg_names(si_1)
    assert ["*a", "**b"] == extract_function_arg_names(si_2)
    assert ["x", "y", "*a", "**b"] == extract_function_arg_names(si_3)
    assert ["x", "y"] == extract_function_arg_names(si_4)
    assert ["x", "y", "*a", "**b"] == extract_function_arg_names(si_5)


def test__safe_invoke():
    assert safe_invoke("si_0") == 1
    assert safe_invoke("si_0", 1, 2, 3, a=1) == 1
    assert safe_invoke("si_1", 1, 2, 5) == 3
    assert safe_invoke("si_1", 1, 2, a=1) == 2
    assert safe_invoke("si_1", 1, 2, 3, b=3, c="d") == 4
    assert safe_invoke("si_1", b=3, a=2) == 5
    assert safe_invoke("si_2", 3, 2) == 5
    assert safe_invoke("si_2", b=3, a=2) == 5
    assert safe_invoke("si_2", 1, 2, 3, b=3, a=2) == 11
    assert safe_invoke("si_3", 1, 2, 3, b=3, a=2) == 10
    assert safe_invoke("si_3", 1, 2, 3, b=3, x=10) == 18
    assert safe_invoke("si_3", y=3, x=10) == 30
    assert safe_invoke("si_4", 6) == 3
    assert safe_invoke("si_4", 6, 3) == 2
    assert safe_invoke("si_4", 10, y=2) == 5
    assert safe_invoke("si_4", a=200, y=3, x=6) == 2
    assert safe_invoke("si_4", 6, x=12) == 2
    assert safe_invoke("si_5", 6) == 60
    assert safe_invoke("si_5", 5, x=12, y=4) == 53
    assert safe_invoke("si_5", **{"x": 8}) == 80
    assert safe_invoke("si_5", 6, 3, 1, 2, a=10) == 31
    assert safe_invoke("si_5", 6, 3, a=10) == 28
    raises(TypeError, lambda: safe_invoke("si_5", a=10))


def si_0():
    return 1


def si_1(a, b):
    return a + b


def si_2(*a, **b):
    return sum(a) + sum(b.values())


def si_3(x, y, *a, **b):
    return x * y + sum(a) + sum(b.values())


def si_4(x, y=2):
    return x / y


def si_5(x, y=10, *a, **b):
    return x * y + sum(a) + sum(b.values())


def efi0():
    pass


def efi1(a: int, b):
    pass


def efi2(a) -> None:
    pass


def efi3(a, b: "int", *arg, **karg) -> "int":
    pass


def efi4(a: int, b: "dummy") -> None:
    pass


def efi5(a: int) -> "dummy":
    pass
