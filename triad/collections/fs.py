from threading import RLock
from typing import Dict, Tuple
from urllib.parse import urlparse
from pathlib import PureWindowsPath

from fs import open_fs, tempfs, memoryfs
from fs.base import FS as FSBase
from fs.mountfs import MountFS
from triad.utils.hash import to_uuid
import os


class FileSystem(MountFS):
    """A unified filesystem based on PyFileSystem2. The special requirement
    for this class is that all paths must be absolute path with scheme.
    To customize different file systems, you should override `create_fs`
    to provide your own configured file systems.
    :Examples:
    >>> fs = FileSystem()
    >>> fs.writetext("mem://from/a.txt", "hello")
    >>> fs.copy("mem://from/a.txt", "mem://to/a.txt")
    :Notice:
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


class _FSPath(object):
    def __init__(self, path: str):
        if path is None:
            raise ValueError("path can't be None")
        path = self._modify_path(path)
        if path.startswith("\\\\") or (
            path[1:].startswith(":\\") and path[0].isalpha()
        ):
            path = PureWindowsPath(path).as_uri()[7:]
            self._scheme = ""
            if path[0] == "/":
                self._root = path[1:4]
                path = path[4:]
            else:
                self._root = "/"
                self.path = path[1:]
            self._path = path.rstrip("/")
        else:
            if path.startswith("file://"):
                path = path[6:]
            if path.startswith("/"):
                self._scheme = ""
                self._root = "/"
                self._path = os.path.abspath(path)
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
    def scheme(self) -> str:
        return self._scheme

    @property
    def root(self) -> str:
        return self._root

    @property
    def relative_path(self) -> str:
        return self._path

    def _modify_path(self, path: str) -> str:
        """to fix things like /s3:/a/b.txt -> s3://a/b.txt"""
        if path.startswith("/"):
            p = path.find("/", 1)
            if p > 1 and path[p - 1] == ":":
                scheme = path[1 : p - 1]
                return scheme + "://" + path[p + 1 :]
        return path
