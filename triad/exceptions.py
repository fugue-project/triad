class NoneArgumentError(ValueError):
    """Exception on None argument
    """

    def __init__(self, message: str):
        super().__init__(message)
