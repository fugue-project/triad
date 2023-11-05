import os
import re
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, List, Tuple

import fsspec
import fsspec.core as fc
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

_SCHEME_PREFIX = re.compile(r"^[a-zA-Z0-9\-_]+:")


@contextmanager
def chdir(path: str) -> Iterator[None]:
    """Change the current working directory to the given path

    :param path: the path to change to

    .. admonition:: Examples

        .. code-block:: python

            from fugue_ml.utils.io import chdir

            with chdir("/tmp"):
                # do something
    """
    op = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(op)


def url_to_fs(path: str, **kwargs: Any) -> Tuple[AbstractFileSystem, str]:
    """A wrapper of ``fsspec.core.url_to_fs``

    :param path: the path to be used
    :param kwargs: additional arguments to ``fsspec.core.url_to_fs``
    :return: the file system and the path
    """
    if path.startswith("file://"):
        path = path[7:]
    return fc.url_to_fs(path, **kwargs)


def exists(path: str) -> bool:
    """Check if a file or a directory exists

    :param path: the path to check
    :return: whether the path (resource) exists
    """
    fs, path = url_to_fs(path)
    return fs.exists(path)


def isdir(path: str) -> bool:
    """Check if a path is a directory

    :param path: the path to check
    :return: whether the path is a directory
    """
    fs, path = url_to_fs(path)
    return fs.isdir(path)


def isfile(path: str) -> bool:
    """Check if a path is a file

    :param path: the path to check
    :return: whether the path is a file
    """
    fs, path = url_to_fs(path)
    return fs.isfile(path)


def abs_path(path: str) -> str:
    """Get the absolute path of a path

    :param path: the path to check
    :return: the absolute path
    """
    p, _path = fc.split_protocol(path)
    if p is None or p == "file":  # local path
        # Path doesn't work with windows
        return os.path.abspath(_path)
    return path


def touch(path: str, auto_mkdir: bool = False) -> None:
    """Create an empty file or update the timestamp of the file

    :param path: the file path
    :param makedirs: if True, create the directory if not exists,
        defaults to False
    """
    fs, _path = url_to_fs(path)
    if auto_mkdir:
        fs.makedirs(fs._parent(_path), exist_ok=True)
    fs.touch(_path, truncate=True)


def rm(path: str, recursive: bool = False) -> None:
    """Remove a file or a directory

    :param path: the path to remove
    :param recursive: if True and the path is directory,
        remove the directory recursively, defaults to False
    """
    fs, path = url_to_fs(path)
    fs.rm(path, recursive=recursive)


def makedirs(path: str, exist_ok: bool = False) -> str:
    """Create a directory

    :param path: the directory path
    :param exist_ok: if True, do not raise error if the directory exists,
        defaults to False

    :return: the absolute directory path
    """
    fs, _path = url_to_fs(path)
    fs.makedirs(_path, exist_ok=exist_ok)
    if isinstance(fs, LocalFileSystem):
        return str(Path(_path).resolve())
    return path


def join(base_path: str, *paths: str) -> str:
    """Join paths with the base path

    :param base_path: the base path
    :param paths: the paths to join to the base path
    :return: the joined path
    """
    if len(paths) == 0:
        return base_path
    p, path = fc.split_protocol(base_path)
    if p is None:  # local path
        return str(Path(base_path).joinpath(*paths))
    return p + "://" + str(PurePosixPath(path).joinpath(*paths))


def glob(path: str) -> List[str]:
    """Glob files

    :param path: the path to glob
    :return: the matched files (absolute paths)
    """
    fs, _path = url_to_fs(path)
    return [fs.unstrip_protocol(x) for x in fs.glob(_path)]


def write_text(path: str, contents: str) -> None:
    """Write text to a file. If the directory of the file does not exist, it
    will create the directory first

    :param path: the file path
    :param contents: the text to write
    """
    with fsspec.open(path, "w") as f:
        f.write(contents)


def read_text(path: str) -> str:
    """Read text from a file

    :param path: the file path
    :return: the text
    """
    with fsspec.open(path, "r") as f:
        return f.read()


def write_bytes(path: str, contents: bytes, create_dir: bool = True) -> None:
    """Write bytes to a file. If the directory of the file does not exist, it
    will create the directory first

    :param path: the file path
    :param contents: the bytes to write
    :param create_dir: if True, create the directory if not exists,
        defaults to True
    """
    fs, path = url_to_fs(path)
    with fs.open(path, "wb") as f:
        f.write(contents)


def read_bytes(path: str) -> bytes:
    """Read bytes from a file

    :param path: the file path
    :return: the bytes
    """
    fs, path = url_to_fs(path)
    with fs.open(path, "rb") as f:
        return f.read()


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
        with fsspec.open(fobj, "wb", create_dir=True) as f:
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
        with fsspec.open(fobj, "rb") as f:
            with unzip_to_temp(f) as tmpdir:
                yield tmpdir
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            with zipfile.ZipFile(fobj, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            yield tmpdirname
