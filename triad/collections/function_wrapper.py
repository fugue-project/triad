import copy
import inspect
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    get_type_hints,
    no_type_check,
)

from ..exceptions import InvalidOperationError
from ..utils.assertion import assert_or_throw
from ..utils.convert import get_full_type_path
from ..utils.entry_points import load_entry_point
from ..utils.hash import to_uuid
from .dict import IndexedOrderedDict


class FunctionWrapper(object):
    _REGISTERED: List[
        Tuple[Type["AnnotatedParam"], Any, str, Callable[[Any], bool]]
    ] = []
    _REGISTERED_CODES: Dict[str, Any] = {}
    _ENTRYPOINT: Optional[str] = None

    def __init__(
        self,
        func: Callable,
        params_re: str = ".*",
        return_re: str = ".*",
    ):
        self._class_method, self._params, self._rt = self._parse_function(
            func, params_re, return_re
        )
        self._func = func

    def __deepcopy__(self, memo: Any) -> Any:
        return copy.copy(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    def __uuid__(self) -> str:
        return to_uuid(get_full_type_path(self._func), self._params, self._rt)

    @property
    def input_code(self) -> str:
        return "".join(x.code for x in self._params.values())

    @property
    def output_code(self) -> str:
        return self._rt.code

    def _parse_function(
        self, func: Callable, params_re: str = ".*", return_re: str = ".*"
    ) -> Tuple[bool, IndexedOrderedDict[str, "AnnotatedParam"], "AnnotatedParam"]:
        sig = inspect.signature(func)
        annotations = get_type_hints(func)
        res: IndexedOrderedDict[str, "AnnotatedParam"] = IndexedOrderedDict()
        class_method = False
        for k, w in sig.parameters.items():
            if k == "self":
                res[k] = SelfParam(w)
                class_method = True
            else:
                anno = annotations.get(k, w.annotation)
                res[k] = self.__class__.parse_annotation(anno, w)
        anno = annotations.get("return", sig.return_annotation)
        rt = self.__class__.parse_annotation(anno, None, none_as_other=False)
        params_str = "".join(x.code for x in res.values())
        assert_or_throw(
            re.match(params_re, params_str) is not None,
            lambda: TypeError(f"Input types not valid {res} for {func}"),
        )
        assert_or_throw(
            re.match(return_re, rt.code) is not None,
            lambda: TypeError(f"Return type not valid {rt} for {func}"),
        )
        return class_method, res, rt

    @no_type_check
    @classmethod
    def annotated_param(  # noqa: C901
        cls,
        annotation: Any,
        code: Optional[str] = None,
        matcher: Optional[Callable[[Any], bool]] = None,
        child_can_reuse_code: bool = False,
    ):
        def _func(tp: Type["AnnotatedParam"]) -> Type["AnnotatedParam"]:
            if not issubclass(tp, AnnotatedParam):
                raise InvalidOperationError(f"{tp} is not a subclass of AnnotatedParam")

            if matcher is not None:
                _matcher = matcher
            else:
                anno = annotation

                def _m(a: Any) -> bool:
                    return a == anno

                _matcher = _m

            tp._annotation = annotation
            if code is not None:
                tp._code = code
            else:
                tp._code = tp.__bases__[0]._code
            if tp._code in cls._REGISTERED_CODES:
                _allow_tp = cls._REGISTERED_CODES[tp._code]
                if (
                    _allow_tp is not None  # implies parent allows reusing the code
                    and inspect.isclass(tp)
                    and issubclass(tp, _allow_tp)
                ):
                    pass
                else:
                    for _ptp, _a, _c, _ in cls._REGISTERED:
                        if _c == tp._code:
                            if str(_ptp) != str(tp):
                                # This is to avoid a cyclic edge case
                                # If the first time import fugue fails, then because
                                # _REGISTERED_CODES is no longer empty, the second call
                                # could re-register the same classes which will cause
                                # exceptions.
                                #
                                # This trick ensures if there were duplication on the
                                # first try then in the second try it still fails at
                                # the same place.
                                #
                                # If import succeeded, this code will never be hit.

                                raise InvalidOperationError(
                                    f"param code {_c} is already registered by {_ptp}"
                                    f" {_a} so can't be used by {tp} {annotation}"
                                )
            else:
                if child_can_reuse_code and inspect.isclass(tp):
                    cls._REGISTERED_CODES[tp._code] = tp
                else:
                    cls._REGISTERED_CODES[tp._code] = None
            cls._REGISTERED.append((tp, annotation, code, _matcher))

            return tp

        return _func

    @classmethod
    def parse_annotation(
        cls,
        annotation: Any,
        param: Optional[inspect.Parameter] = None,
        none_as_other: bool = True,
    ) -> "AnnotatedParam":
        if annotation == type(None):  # noqa: E721
            return OtherParam(param) if none_as_other else NoneParam(param)
        if annotation == inspect.Parameter.empty:
            if param is not None and param.kind == param.VAR_POSITIONAL:
                return PositionalParam(param)
            if param is not None and param.kind == param.VAR_KEYWORD:
                return KeywordParam(param)
            return OtherParam(param) if none_as_other else NoneParam(param)

        load_entry_point(cls._ENTRYPOINT)

        for tp, _, _, matcher in cls._REGISTERED:
            if matcher(annotation):
                return tp(param)

        if param is not None and param.kind == param.VAR_POSITIONAL:
            return PositionalParam(param)
        if param is not None and param.kind == param.VAR_KEYWORD:
            return KeywordParam(param)
        return OtherParam(param)


class AnnotatedParam:
    def __init__(self, param: Optional[inspect.Parameter]):
        if param is not None:
            self.required = param.default == inspect.Parameter.empty
            self.default = param.default
        else:
            self.required, self.default = True, None
        self.annotation: Any = getattr(self.__class__, "_annotation")  # noqa
        self.code: str = getattr(self.__class__, "_code")  # noqa

    def __repr__(self) -> str:
        return str(self.annotation)


def function_wrapper(entrypoint: Optional[str]):
    def _func(tp: Type[FunctionWrapper]) -> Type[FunctionWrapper]:
        if not issubclass(tp, FunctionWrapper):
            raise InvalidOperationError(f"{tp} is not a subclass of FunctionWrapper")

        setattr(tp, "_REGISTERED", list(FunctionWrapper._REGISTERED))  # noqa
        setattr(  # noqa
            tp, "_REGISTERED_CODES", dict(FunctionWrapper._REGISTERED_CODES)
        )
        setattr(tp, "_ENTRYPOINT", entrypoint)  # noqa

        return tp

    return _func


@FunctionWrapper.annotated_param("NoneType", "n", lambda a: False)
class NoneParam(AnnotatedParam):
    pass


@FunctionWrapper.annotated_param("[Self]", "0", lambda a: False)
class SelfParam(AnnotatedParam):
    pass


@FunctionWrapper.annotated_param("[Other]", "x", lambda a: False)
class OtherParam(AnnotatedParam):
    pass


@FunctionWrapper.annotated_param("[Positional]", "y", lambda a: False)
class PositionalParam(AnnotatedParam):
    pass


@FunctionWrapper.annotated_param("[Keyword]", "z", lambda a: False)
class KeywordParam(AnnotatedParam):
    pass
