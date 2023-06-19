import os
from io import BytesIO

from pytest import raises

import triad.utils.io as iou
from triad.utils.io import _modify_path


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


def test_modify_path():
    assert "c:/" == _modify_path("/c:")
    assert "s3://" == _modify_path("/s3:")
    assert "C:/" == _modify_path("/C:\\")
    assert "C:/a" == _modify_path("/C:\\a")
    assert "C:/" == _modify_path("/C:\\\\")
    assert "C:/a/b" == _modify_path("/C:\\\\a\\b")
    assert "C:/" == _modify_path("/C:/")
    assert "C:/a" == _modify_path("/C:/a")
    assert "C:/" == _modify_path("/C://")
    assert "C:/a/b" == _modify_path("/C://a/b")

    assert "/" == _modify_path("file://")
    assert "/a/b" == _modify_path("file://a/b")
    assert "c:/a/b" == _modify_path("file:///c:/a/b")
    assert "C:/" == _modify_path("C://")
    assert "c:/x" == _modify_path("c://x")
    assert "c:/" == _modify_path("c:/")
    assert "c:/x" == _modify_path("c:/x")
    assert "c:/" == _modify_path("c:")
    assert "c:/" == _modify_path("c:\\")
    assert "c:/x/" == _modify_path("c:\\x\\")
    raises(NotImplementedError, lambda: _modify_path("\\\\10.0.0.1\1"))
