import os
from io import BytesIO

import pytest
import sys
import os

import triad.utils.io as iou


def test_join(tmpdir):
    assert iou.join(str(tmpdir)) == os.path.join(str(tmpdir))
    assert iou.join(str(tmpdir), "a", "b") == os.path.join(str(tmpdir), "a", "b")
    assert iou.join("dummy://", "a", "b", "c/") == "dummy://a/b/c"
    assert iou.join("dummy://a/", "b/", "c/") == "dummy://a/b/c"
    assert iou.join("dummy://a/", "b/", "*.parquet") == "dummy://a/b/*.parquet"


def test_glob(tmpdir):
    assert iou.glob(os.path.join(str(tmpdir), "a")) == []
    assert iou.glob(os.path.join(str(tmpdir), "a", "*.txt")) == []
    iou.touch(os.path.join(str(tmpdir), "a.txt"))
    assert iou.glob(os.path.join(str(tmpdir), "a.txt")) == [
        os.path.join(str(tmpdir), "a.txt")
    ]
    assert iou.glob(os.path.join(str(tmpdir), "*.txt")) == [
        os.path.join(str(tmpdir), "a.txt")
    ]
    iou.touch(os.path.join(str(tmpdir), "a", "a.txt"), auto_mkdir=True)
    assert iou.glob(os.path.join(str(tmpdir), "a", "*.txt")) == [
        os.path.join(str(tmpdir), "a", "a.txt")
    ]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="not a test for windows")
def test_join_not_win():
    assert iou.join("a", "b", "c/") == "a/b/c"
    assert iou.join("/a", "b", "c/") == "/a/b/c"
    assert iou.join("/a", "b", "*.parquet") == "/a/b/*.parquet"


@pytest.mark.skipif(
    not sys.platform.startswith("win"), reason="a test only for windows"
)
def test_join_is_win():
    assert iou.join("a", "b", "c") == "a\\b\\c"
    assert iou.join("c:\\a", "b", "c") == "c:\\a\\b\\c"
    assert iou.join("c:\\a", "b", "*.parquet") == "c:\\a\\b\\*.parquet"


def test_makedirs(tmpdir):
    path = os.path.join(str(tmpdir), "temp", "a")
    assert path == iou.makedirs(path, exist_ok=False)
    assert iou.exists(path)
    with pytest.raises(OSError):
        iou.makedirs(path, exist_ok=False)
    iou.makedirs(path, exist_ok=True)

    op = os.getcwd()
    os.chdir(str(tmpdir))
    assert os.path.join(str(tmpdir), "temp", "b") == iou.makedirs(
        iou.join("temp", "b"), exist_ok=False
    )
    assert iou.exists(iou.join("temp", "b"))
    assert iou.exists(os.path.join(str(tmpdir), "temp", "b"))
    os.chdir(op)

    path = "memory://temp/a"
    assert path == iou.makedirs(path, exist_ok=True)
    assert iou.exists(path)
    with pytest.raises(OSError):
        iou.makedirs(path, exist_ok=False)


def test_exists(tmpdir):
    iou.write_text(os.path.join(str(tmpdir), "a.txt"), "a")
    assert iou.exists(os.path.join(str(tmpdir), "a.txt"))
    assert not iou.exists(os.path.join(str(tmpdir), "b.txt"))
    iou.write_text(os.path.join(str(tmpdir), "temp", "a.txt"), "a")
    assert iou.exists(os.path.join(str(tmpdir), "temp", "a.txt"))
    assert iou.exists(os.path.join(str(tmpdir), "temp"))
    assert iou.exists(str(tmpdir))


def test_is_dir_or_file(tmpdir):
    for base in [str(tmpdir), "memory://"]:
        path = os.path.join(base, "a.txt")
        assert not iou.isfile(path)
        assert not iou.isdir(path)
        iou.write_text(path, "a")
        assert iou.isfile(path)
        assert not iou.isdir(path)
        path = os.path.join(base, "b")
        iou.makedirs(path)
        assert not iou.isfile(path)
        assert iou.isdir(path)


def test_touch(tmpdir):
    for base in [str(tmpdir), "memory://"]:
        path = os.path.join(base, "a.txt")
        iou.touch(path)
        assert iou.isfile(path)

        iou.write_text(path, "a")
        assert iou.read_text(path) == "a"
        iou.touch(path)
        assert iou.isfile(path)
        assert iou.read_text(path) == ""

        if not base.startswith("memory"):
            pytest.raises(OSError, lambda: iou.touch(iou.join(base, "b", "c.txt")))

        iou.touch(iou.join(base, "b", "c.txt"), auto_mkdir=True)
        iou.exists(iou.join(base, "b", "c.txt"))


def test_rm(tmpdir):
    path = os.path.join(str(tmpdir), "a.txt")
    for recursive in [True, False]:
        iou.write_text(os.path.join(str(tmpdir), "a.txt"), "a")
        assert iou.exists(path)
        iou.rm(path, recursive=recursive)
        assert not iou.exists(path)

    path = os.path.join(str(tmpdir), "b", "c")
    iou.write_text(os.path.join(path, "a.txt"), "a")
    assert iou.exists(path)
    iou.rm(path, recursive=True)
    assert not iou.exists(path)


def test_read_write_content(tmpdir):
    iou.write_text(os.path.join(str(tmpdir), "temp", "a.txt"), "aa")
    assert iou.read_text(os.path.join(str(tmpdir), "temp", "a.txt")) == "aa"
    iou.write_bytes(os.path.join(str(tmpdir), "temp", "a.bin"), b"aa")
    assert iou.read_bytes(os.path.join(str(tmpdir), "temp", "a.bin")) == b"aa"


def test_zip_unzip(tmpdir):
    bio = BytesIO()
    with iou.zip_temp(bio) as tmpdir:
        iou.write_text(os.path.join(tmpdir, "a.txt"), "a")
        iou.write_text(os.path.join(tmpdir, "b.txt"), "b")

    with iou.unzip_to_temp(BytesIO(bio.getvalue())) as tmpdir:
        assert iou.read_text(os.path.join(tmpdir, "a.txt")) == "a"
        assert iou.read_text(os.path.join(tmpdir, "b.txt")) == "b"

    tf = os.path.join(str(tmpdir), "temp", "x.zip")
    with iou.zip_temp(tf) as tmpdir:
        iou.write_text(os.path.join(tmpdir, "a.txt"), "a")
        iou.write_text(os.path.join(tmpdir, "b.txt"), "b")

    with iou.unzip_to_temp(tf) as tmpdir:
        assert iou.read_text(os.path.join(tmpdir, "a.txt")) == "a"
        assert iou.read_text(os.path.join(tmpdir, "b.txt")) == "b"
