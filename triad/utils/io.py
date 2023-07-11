import os
import re
import tempfile
import zipfile
from contextlib import contextmanager
from typing import Any, Iterator

import fsspec

_SCHEME_PREFIX = re.compile(r"^[a-zA-Z0-9\-_]+:")


def exists(path: str) -> bool:
    """Check if a file or a directory exists

    :param path: the path to check
    :return: whether the path (resource) exists
    """
    fs, path = fsspec.core.url_to_fs(path)
    return fs.exists(path)


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
    fs, path = fsspec.core.url_to_fs(path)
    with fs.open(path, "wb") as f:
        f.write(contents)


def read_bytes(path: str) -> bytes:
    """Read bytes from a file

    :param path: the file path
    :return: the bytes
    """
    fs, path = fsspec.core.url_to_fs(path)
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
