def is_quoteless_column_name(expr: str) -> bool:
    """Check if `expr` can be used without ``

    :param expr: column name expression
    :return: whether it is valid
    """
    return isinstance(expr, str) and expr.isidentifier() and expr.isascii()
