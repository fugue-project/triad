import os
import tempfile
import zipfile
from contextlib import contextmanager
from typing import Any, Iterator, IO
import fs


@contextmanager
def open(  # pylint: disable=W0622 # noqa: A001
    path: str, mode: str, create_dir: bool = False, **kwargs: Any
) -> Iterator[IO]:
    dirname = fs.path.dirname(path)
    basename = fs.path.basename(path)
    tfs = fs.open_fs(dirname, create=create_dir)
    with tfs.open(basename, mode, **kwargs) as f:
        yield f


def write_text(path: str, contents: str, create_dir: bool = True):
    dirname = fs.path.dirname(path)
    basename = fs.path.basename(path)
    tfs = fs.open_fs(dirname, writeable=True, create=create_dir)
    tfs.writetext(basename, contents)


def read_text(path: str) -> str:
    dirname = fs.path.dirname(path)
    basename = fs.path.basename(path)
    tfs = fs.open_fs(dirname)
    return tfs.readtext(basename)


def write_bytes(path: str, contents: bytes, create_dir: bool = True):
    dirname = fs.path.dirname(path)
    basename = fs.path.basename(path)
    tfs = fs.open_fs(dirname, writeable=True, create=create_dir)
    tfs.writebytes(basename, contents)


def read_bytes(path: str) -> bytes:
    dirname = fs.path.dirname(path)
    basename = fs.path.basename(path)
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
        dirname = fs.path.dirname(fobj)
        basename = fs.path.basename(fobj)
        tfs = fs.open_fs(dirname, writeable=True, create=True)
        with tfs.open(basename, "wb") as f:
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
        dirname = fs.path.dirname(fobj)
        basename = fs.path.basename(fobj)
        tfs = fs.open_fs(dirname)
        with tfs.open(basename, "rb") as f:
            with unzip_to_temp(f) as tmpdir:
                yield tmpdir
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            with zipfile.ZipFile(fobj, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            yield tmpdirname
