from threading import RLock
from typing import Dict, Tuple
from urllib.parse import urlparse

from fs import open_fs, tempfs
from fs.base import FS as FSBase
from fs.mountfs import MountFS
from triad.utils.hash import to_uuid
import os


class FSPath(object):
    def __init__(self, path: str):
        if path is None:
            raise ValueError("path can't be None")
        path = self._modify_path(path)
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
        """to fix things like /s3:/a/b.txt -> s3://a/b.txt
        """
        if path.startswith("/"):
            p = path.find("/", 1)
            if p > 1 and path[p - 1] == ":":
                scheme = path[1 : p - 1]
                return scheme + "://" + path[p + 1 :]
        return path


class FileSystem(MountFS):
    def __init__(self, auto_close: bool = True):
        super().__init__(auto_close)
        self._fs_store: Dict[str, FSBase] = {}
        self._in_create = False
        self._fs_lock = RLock()

    def create_fs(self, root: str) -> FSBase:
        if root.startswith("temp://"):
            fs = tempfs.TempFS(root[7:])
            return fs
        return open_fs(root)

    def _delegate(self, path) -> Tuple[FSBase, str]:
        with self._fs_lock:
            if self._in_create:  # pragma: no cover
                return super()._delegate(path)
            self._in_create = True
            fp = FSPath(path)
            if fp.root not in self._fs_store:
                self._fs_store[fp.root] = self.create_fs(fp.root)
                self.mount(to_uuid(fp.root), self._fs_store[fp.root])
            self._in_create = False
        m_path = os.path.join(to_uuid(fp.root), fp.relative_path)
        return super()._delegate(m_path)
