import os
import re
import tempfile
import zipfile
from contextlib import contextmanager
from typing import IO, Any, Iterator, Tuple

import fs

_SCHEME_PREFIX = re.compile(r"^[a-zA-Z0-9\-_]+:")


@contextmanager
def open_file(
    path: str, mode: str, create_dir: bool = False, **kwargs: Any
) -> Iterator[IO]:
    dirname, basename = _split_dir_file(path)
    tfs = fs.open_fs(dirname, create=create_dir)
    with tfs.open(basename, mode, **kwargs) as f:
        yield f


def write_text(path: str, contents: str, create_dir: bool = True):
    dirname, basename = _split_dir_file(path)
    tfs = fs.open_fs(dirname, writeable=True, create=create_dir)
    tfs.writetext(basename, contents)


def read_text(path: str) -> str:
    dirname, basename = _split_dir_file(path)
    tfs = fs.open_fs(dirname)
    return tfs.readtext(basename)


def write_bytes(path: str, contents: bytes, create_dir: bool = True):
    dirname, basename = _split_dir_file(path)
    tfs = fs.open_fs(dirname, writeable=True, create=create_dir)
    tfs.writebytes(basename, contents)


def read_bytes(path: str) -> bytes:
    dirname, basename = _split_dir_file(path)
    tfs = fs.open_fs(dirname)
    return tfs.readbytes(basename)


@contextmanager
def zip_temp(fobj: Any) -> Iterator[str]:
    """Zip a temporary directory to a file object.

    :param fobj: the file path or file object

    .. admonition:: Examples

        .. code-block:: python

            from fugue_ml.utils.io import zip_temp
            from io import BytesIO

            bio = BytesIO()
            with zip_temp(bio) as tmpdir:
                # do something with tmpdir (string)
    """
    if isinstance(fobj, str):
        with open_file(fobj, "wb", create_dir=True) as f:
            with zip_temp(f) as tmpdir:
                yield tmpdir
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname

            with zipfile.ZipFile(
                fobj, "w", zipfile.ZIP_DEFLATED, allowZip64=True
            ) as zf:
                for root, _, filenames in os.walk(tmpdirname):
                    for name in filenames:
                        file_path = os.path.join(root, name)
                        rel_dir = os.path.relpath(root, tmpdirname)
                        rel_name = os.path.normpath(os.path.join(rel_dir, name))
                        zf.write(file_path, rel_name)


@contextmanager
def unzip_to_temp(fobj: Any) -> Iterator[str]:
    """Unzip a file object into a temporary directory.

    :param fobj: the file object

    .. admonition:: Examples

        .. code-block:: python

            from fugue_ml.utils.io import zip_temp
            from io import BytesIO

            bio = BytesIO()
            with zip_temp(bio) as tmpdir:
                # create files in the tmpdir (string)

            with unzip_to_temp(BytesIO(bio.getvalue())) as tmpdir:
                # read files from the tmpdir (string)
    """
    if isinstance(fobj, str):
        with open_file(fobj, "rb") as f:
            with unzip_to_temp(f) as tmpdir:
                yield tmpdir
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            with zipfile.ZipFile(fobj, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            yield tmpdirname


def _split_dir_file(path: str) -> Tuple[str, str]:
    path = _modify_path(path)
    dirname = fs.path.dirname(path)
    basename = fs.path.basename(path)
    return dirname, basename


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
