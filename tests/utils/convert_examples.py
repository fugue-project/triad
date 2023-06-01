from triad.utils.convert import to_type


class BaseClass:
    pass


class SubClass(BaseClass):
    pass


class Class2:
    def __init__(self, a=0, b=0, c=0):
        self.s = str(a) + str(b) + str(c)


class __Dummy__:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def invoke_to_type(exp):
    return to_type(exp)
