import os
from io import BytesIO

import pytest
import sys

import triad.utils.io as iou


def test_join(tmpdir):
    assert iou.join(str(tmpdir)) == os.path.join(str(tmpdir))
    assert iou.join(str(tmpdir), "a", "b") == os.path.join(str(tmpdir), "a", "b")
    assert iou.join("dummy://", "a", "b", "c/") == "dummy://a/b/c"
    assert iou.join("dummy://a/", "b/", "c/") == "dummy://a/b/c"
    assert iou.join("dummy://a/", "b/", "*.parquet") == "dummy://a/b/*.parquet"


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
    iou.makedirs(os.path.join(str(tmpdir), "temp", "a"), exist_ok=False)
    assert iou.exists(os.path.join(str(tmpdir), "temp", "a"))
    with pytest.raises(OSError):
        iou.makedirs(os.path.join(str(tmpdir), "temp", "a"), exist_ok=False)
    iou.makedirs(os.path.join(str(tmpdir), "temp", "a"), exist_ok=True)

    iou.makedirs("memory://temp/a", exist_ok=True)
    assert iou.exists("memory://temp/a")
    with pytest.raises(OSError):
        iou.makedirs("memory://temp/a", exist_ok=False)


def test_exists(tmpdir):
    iou.write_text(os.path.join(str(tmpdir), "a.txt"), "a")
    assert iou.exists(os.path.join(str(tmpdir), "a.txt"))
    assert not iou.exists(os.path.join(str(tmpdir), "b.txt"))
    iou.write_text(os.path.join(str(tmpdir), "temp", "a.txt"), "a")
    assert iou.exists(os.path.join(str(tmpdir), "temp", "a.txt"))
    assert iou.exists(os.path.join(str(tmpdir), "temp"))
    assert iou.exists(str(tmpdir))


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
