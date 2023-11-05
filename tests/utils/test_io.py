import os
import sys
from io import BytesIO

import pytest

import triad.utils.io as iou


def test_join(tmpdir):
    assert iou.join(str(tmpdir)) == os.path.join(str(tmpdir))
    assert iou.join(str(tmpdir), "a", "b") == os.path.join(str(tmpdir), "a", "b")
    assert iou.join("dummy://", "a", "b", "c/") == "dummy://a/b/c"
    assert iou.join("dummy://a/", "b/", "c/") == "dummy://a/b/c"
    assert iou.join("dummy://a/", "b/", "*.parquet") == "dummy://a/b/*.parquet"


def test_glob(tmpdir):
    def _assert(gres, expected):
        for a, b in zip(gres, expected):
            assert iou.exists(a) and iou.exists(b)

    assert iou.glob(os.path.join(str(tmpdir), "a")) == []
    assert iou.glob(os.path.join(str(tmpdir), "a", "*.txt")) == []
    iou.touch(os.path.join(str(tmpdir), "a.txt"))
    _assert(
        iou.glob(os.path.join(str(tmpdir), "a.txt")),
        [os.path.join(str(tmpdir), "a.txt")],
    )
    _assert(
        iou.glob(os.path.join(str(tmpdir), "*.txt")),
        [os.path.join(str(tmpdir), "a.txt")],
    )
    iou.touch(os.path.join(str(tmpdir), "a", "a.txt"), auto_mkdir=True)
    _assert(
        iou.glob(os.path.join(str(tmpdir), "a", "*.txt")),
        [os.path.join(str(tmpdir), "a", "a.txt")],
    )

    iou.touch("memory://gtest/m.txt", auto_mkdir=True)
    assert iou.glob("memory://gtest/*.txt") == ["memory:///gtest/m.txt"]


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


@pytest.mark.skipif(sys.platform.startswith("win"), reason="not a test for windows")
def test_abs_path_not_win(tmpdir):
    with iou.chdir(str(tmpdir)):
        assert iou.abs_path("a") == os.path.join(str(tmpdir), "a")
    assert iou.abs_path("/tmp/x") == "/tmp/x"
    assert iou.abs_path("file:///tmp/x") == "/tmp/x"
    assert iou.abs_path("memory://tmp/x") == "memory://tmp/x"
    assert iou.abs_path("memory:///tmp/x") == "memory:///tmp/x"
    assert iou.abs_path("dummy:///tmp/x") == "dummy:///tmp/x"


@pytest.mark.skipif(
    not sys.platform.startswith("win"), reason="a test only for windows"
)
def test_abs_path_is_win(tmpdir):
    with iou.chdir(str(tmpdir)):
        assert iou.abs_path("a") == os.path.join(str(tmpdir), "a")
    assert iou.abs_path("c:\\tmp\\x") == "c:\\tmp\\x"
    assert iou.abs_path("file://c:/tmp/x") == "c:\\tmp\\x"
    assert iou.abs_path("memory://tmp/x") == "memory://tmp/x"
    assert iou.abs_path("memory:///tmp/x") == "memory:///tmp/x"


def test_makedirs(tmpdir):
    path = os.path.join(str(tmpdir), "temp", "a")
    assert path == iou.makedirs(path, exist_ok=False)
    assert iou.exists(path)
    with pytest.raises(OSError):
        iou.makedirs(path, exist_ok=False)
    iou.makedirs(path, exist_ok=True)

    with iou.chdir(str(tmpdir)):
        assert os.path.join(str(tmpdir), "temp", "b") == iou.makedirs(
            iou.join("temp", "b"), exist_ok=False
        )
        assert iou.exists(iou.join("temp", "b"))
        assert iou.exists(os.path.join(str(tmpdir), "temp", "b"))

    path = "memory://temp/a"
    assert path == iou.makedirs(path, exist_ok=True)
    assert iou.exists(path)
    with pytest.raises(OSError):
        iou.makedirs(path, exist_ok=False)

    path = os.path.join(str(tmpdir), "temp", "aa")
    fs, _path = iou.url_to_fs(path)
    _path = fs.unstrip_protocol(_path)  # file:///...
    iou.makedirs(_path, exist_ok=False)
    assert iou.exists(path)


def test_exists(tmpdir):
    iou.write_text(os.path.join(str(tmpdir), "a.txt"), "a")
    assert iou.exists(os.path.join(str(tmpdir), "a.txt"))
    assert not iou.exists(os.path.join(str(tmpdir), "b.txt"))
    iou.write_text(os.path.join(str(tmpdir), "temp", "a.txt"), "a")
    assert iou.exists(os.path.join(str(tmpdir), "temp", "a.txt"))
    assert iou.exists(os.path.join(str(tmpdir), "temp"))
    assert iou.exists(str(tmpdir))

    with iou.chdir(str(tmpdir)):
        assert iou.exists("a.txt")
        assert iou.exists(os.path.join("temp", "a.txt"))
        assert iou.exists("temp")
        assert iou.exists(".")


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
