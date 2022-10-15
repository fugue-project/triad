import logging
import sys
from typing import Any, List

from .threading import run_once

if sys.version_info >= (3, 8):
    from importlib.metadata import EntryPoint, entry_points

    _IMPORTLIB_META_VERSION: Any = ()
else:  # pragma: no cover
    from importlib_metadata import EntryPoint, entry_points, version

    _IMPORTLIB_META_VERSION = tuple(
        int(i) for i in version("importlib_metadata").split(".")[:2]
    )


@run_once(key_func=lambda name: name)
def load_entry_point(name: str) -> None:
    """Load dependencies and functions of a given entrypoint. For any
    given entrypoint name, it will be loaded only once in one process.

    :param name: the name of the entrypoint

    .. admonition:: Example

        Assume in ``setup.py``, you have:

        .. code-block:: python

            setup(
                ...,
                entry_points={
                    "my.plugins": [
                        "my = pkg2.module2"
                    ]
                },
            )

        And this is how you load ``my.plugins``:

        .. code-block:: python

            from triad.utils.entrypoints import load_entry_point

            load_entry_point("my.plugins")
    """
    for plugin in _entry_points_for(name):
        _load_plugin(name, plugin)


def _load_plugin(entrypoint: str, plugin: EntryPoint) -> None:
    logger = logging.getLogger(_load_plugin.__name__)
    try:
        res = plugin.load()
        if callable(res):
            res()
        logger.debug("loaded %s %s", entrypoint, plugin)
    except Exception as e:  # pragma: no cover
        logger.debug("failed to load %s %s: %s", entrypoint, plugin, e)


def _entry_points_for(key: str) -> List[EntryPoint]:  # pragma: no cover
    if sys.version_info >= (3, 10) or _IMPORTLIB_META_VERSION >= (3, 6):
        return entry_points().select(group=key)  # type: ignore
    else:
        return entry_points().get(key, {})  # type: ignore
