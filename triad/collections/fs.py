import re
from threading import RLock
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

from triad.utils.hash import to_uuid

import fs
from fs import memoryfs, open_fs, tempfs
from fs.base import FS as FSBase
from fs.glob import BoundGlobber, Globber
from fs.mountfs import MountFS
from fs.subfs import SubFS

_SCHEME_PREFIX = re.compile(r"^[a-zA-Z0-9\-_]+:")


class FileSystem(MountFS):
    """A unified filesystem based on PyFileSystem2. The special requirement
    for this class is that all paths must be absolute path with scheme.
    To customize different file systems, you should override `create_fs`
    to provide your own configured file systems.

    .. admonition:: Examples

        .. code-block:: python

            fs = FileSystem()
            fs.writetext("mem://from/a.txt", "hello")
            fs.copy("mem://from/a.txt", "mem://to/a.txt")

    .. note::

        If a path is not a local path, it must include the scheme and `netloc`
        (the first element after `://`)
        :param auto_close: If `True` (the default), the child filesystems
        will be closed when `MountFS` is closed.
    """

    def __init__(self, auto_close: bool = True):
        super().__init__(auto_close)
        self._fs_store: Dict[str, FSBase] = {}
        self._in_create = False
        self._fs_lock = RLock()

    def create_fs(self, root: str) -> FSBase:
        """create a PyFileSystem instance from `root`. `root` is in the
        format of `/` if local path, else `<scheme>://<netloc>`.
        You should override this method to provide custom instances, for
        example, if you want to create an S3FS with certain parameters.
        :param root: `/` if local path, else `<scheme>://<netloc>`
        """
        if root.startswith("temp://"):
            fs = tempfs.TempFS(root[len("temp://") :])
            return fs
        if root.startswith("mem://"):
            fs = memoryfs.MemoryFS()
            return fs
        return open_fs(root)

    @property
    def glob(self):
        """A globber object"""
        return _BoundGlobber(self)

    def _delegate(self, path) -> Tuple[FSBase, str]:
        with self._fs_lock:
            if self._in_create:  # pragma: no cover
                return super()._delegate(path)
            self._in_create = True
            fp = _FSPath(path)
            if fp.root not in self._fs_store:
                self._fs_store[fp.root] = self.create_fs(fp.root)
                self.mount(to_uuid(fp.root), self._fs_store[fp.root])
            self._in_create = False
        m_path = to_uuid(fp.root) + "/" + fp.relative_path
        return super()._delegate(m_path)

    def makedirs(
        self, path: str, permissions: Any = None, recreate: bool = False
    ) -> SubFS:
        """Make a directory, and any missing intermediate directories.

        .. note::

            This overrides the base ``makedirs``

        :param path: path to directory from root.
        :param permissions: initial permissions, or `None` to use defaults.
        :recreate: if `False` (the default), attempting to
          create an existing directory will raise an error. Set
          to `True` to ignore existing directories.
        :return: a sub-directory filesystem.

        :raises fs.errors.DirectoryExists: if the path is already
          a directory, and ``recreate`` is `False`.
        :raises fs.errors.DirectoryExpected: if one of the ancestors
          in the path is not a directory.
        """
        self.check()
        fs, _path = self._delegate(path)
        return fs.makedirs(_path, permissions=permissions, recreate=recreate)


class _BoundGlobber(BoundGlobber):
    def __call__(
        self,
        pattern: Any,
        path: str = "/",
        namespaces: Any = None,
        case_sensitive: bool = True,
        exclude_dirs: Any = None,
    ) -> Globber:
        fp = _FSPath(path)
        _path = fs.path.join(fp._root, fp._path) if fp.is_windows else path
        return super().__call__(
            pattern,
            path=_path,
            namespaces=namespaces,
            case_sensitive=case_sensitive,
            exclude_dirs=exclude_dirs,
        )


class _FSPath(object):
    def __init__(self, path: str):
        if path is None:
            raise ValueError("path can't be None")
        path = _modify_path(path)
        self._is_windows = False
        if _is_windows(path):
            self._scheme = ""
            self._root = path[:3]
            self._path = path[3:]
            self._is_windows = True
        elif path.startswith("/"):
            self._scheme = ""
            self._root = "/"
            self._path = fs.path.abspath(path)
        else:
            uri = urlparse(path)
            if uri.scheme == "" and not path.startswith("/"):
                raise ValueError(
                    f"invalid {path}, must be abs path either local or with scheme"
                )
            self._scheme = uri.scheme
            if uri.netloc == "":
                raise ValueError(f"invalid path {path}")
            self._root = uri.scheme + "://" + uri.netloc
            self._path = uri.path
        self._path = self._path.lstrip("/")
        # if self._path == "":
        #    raise ValueError(f"invalid path {path}")

    @property
    def is_windows(self) -> bool:
        return self._is_windows

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def root(self) -> str:
        return self._root

    @property
    def relative_path(self) -> str:
        return self._path


def _modify_path(path: str) -> str:  # noqa: C901
    """to fix things like /s3:/a/b.txt -> s3://a/b.txt"""
    if path.startswith("/"):
        s = _SCHEME_PREFIX.search(path[1:])
        if s is not None:
            colon = s.end()
            scheme = path[1:colon]
            if colon + 1 == len(path):  # /C: or /s3:
                path = scheme + "://"
            elif path[colon + 1] == "/":  # /s3:/a/b.txt
                path = scheme + "://" + path[colon + 1 :].lstrip("/")
            elif path[colon + 1] == "\\":  # /c:\a\b.txt
                path = scheme + ":\\" + path[colon + 1 :].lstrip("\\")
    if path.startswith("file:///"):
        path = path[8:]
    elif path.startswith("file://"):
        path = path[6:]
    if path.startswith("\\\\"):
        # windows \\10.100.168.1\...
        raise NotImplementedError(f"path {path} is not supported")
    if path != "" and path[0].isalpha():
        if len(path) == 2 and path[1] == ":":
            # C: => C:/
            return path[0] + ":/"
        if path[1:].startswith(":\\"):
            # C:\a\b\c => C:/a/b/c
            return path[0] + ":/" + path[3:].replace("\\", "/").lstrip("/")
        if path[1:].startswith(":/"):
            # C:/a/b/c => C:/a/b/c
            return path[0] + ":/" + path[3:].lstrip("/")
    return path


def _is_windows(path: str) -> bool:
    if len(path) < 3:
        return False
    return path[0].isalpha() and path[1] == ":" and path[2] == "/"
