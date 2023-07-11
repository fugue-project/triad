import os
from io import BytesIO

from pytest import raises

import triad.utils.io as iou


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
