def validate_triad_var_name(expr: str) -> bool:
    """Check if `expr` is a valid Triad variable name based on Triad standard:
    it has to be a valid python identifier and it can't be purely `_`

    :param expr: column name expression
    :return: whether it is valid
    """
    if not isinstance(expr, str) or not expr.isidentifier() or not expr.isascii():
        return False
    return expr.strip("_") != ""


def assert_triad_var_name(expr: str) -> str:
    """Check if `expr` is a valid Triad variable name based on Triad standard:
    it has to be a valid python identifier and it can't be purely `_`

    :param expr: column name expression
    :raises AssertionError: if the expression is invalid
    :return: the expression string
    """
    if validate_triad_var_name(expr):
        return expr
    raise AssertionError(f"{expr} is not a valid Triad variable name")
