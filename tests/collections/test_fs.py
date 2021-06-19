import fs as pfs
from os.path import exists
import os

from pytest import raises
from triad.collections.fs import FileSystem, _FSPath, _modify_path, _is_windows


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


def test_is_windows():
    assert not _is_windows("")
    assert not _is_windows("c")
    assert not _is_windows("c:")
    assert not _is_windows("c:\\")
    assert _is_windows("c:/")
    assert _is_windows("c://")
    assert _is_windows("c:/x")


def test__FSPath():
    p = _FSPath("/a/b.txt")
    assert "" == p.scheme
    assert "/" == p.root
    assert "a/b.txt" == p.relative_path

    p = _FSPath("//a.txt")
    assert "" == p.scheme
    assert "/" == p.root
    assert "a.txt" == p.relative_path

    p = _FSPath("file://a/b.txt")
    assert "" == p.scheme
    assert "/" == p.root
    assert "a/b.txt" == p.relative_path

    p = _FSPath("hdfs://a/b.txt")
    assert "hdfs" == p.scheme
    assert "hdfs://a" == p.root
    assert "b.txt" == p.relative_path

    p = _FSPath("temp://a/b")
    assert "temp" == p.scheme
    assert "temp://a" == p.root
    assert "b" == p.relative_path

    p = _FSPath("/temp:/a/b")
    assert "temp" == p.scheme
    assert "temp://a" == p.root
    assert "b" == p.relative_path
    assert not p.is_windows

    # Windows test cases
    p = _FSPath("c:\\folder\\myfile.txt")
    assert "" == p.scheme
    assert "c:/" == p.root
    assert "folder/myfile.txt" == p.relative_path
    assert p.is_windows

    p = _FSPath("c://folder/myfile.txt")
    assert "" == p.scheme
    assert "c:/" == p.root
    assert "folder/myfile.txt" == p.relative_path
    assert p.is_windows

    p = _FSPath("c:/folder/myfile.txt")
    assert "" == p.scheme
    assert "c:/" == p.root
    assert "folder/myfile.txt" == p.relative_path
    assert p.is_windows

    raises(
        NotImplementedError,
        lambda: _FSPath("\\\\123.123.123.123\\share\\folder\\myfile.txt"),
    )

    raises(ValueError, lambda: _FSPath(None))
    raises(ValueError, lambda: _FSPath(""))
    raises(ValueError, lambda: _FSPath("a.txt"))
    raises(ValueError, lambda: _FSPath("hdfs://"))


def test_fs(tmpdir):
    tmpdir = str(tmpdir)
    # Tests to read and write with tmpdir without FS
    tmpfile = os.path.join(tmpdir, "f.txt")
    f = open(tmpfile, "a")
    f.write("read test")
    f.close()
    f = open(tmpfile, "r")
    assert f.read() == "read test"

    p1 = os.path.join(tmpdir, "a")
    p2 = os.path.join(tmpdir, "b")
    assert not exists(p1)
    assert not exists(p2)
    fs = MockFS()
    fs.makedirs(p1)
    fs.makedirs(p2)
    assert fs.exists(p1) and exists(p1) and os.path.isdir(p1)
    assert fs.exists(p2) and exists(p2) and os.path.isdir(p2)
    assert 1 == fs.create_called
    fs.create_called = 0
    fs.makedirs("temp://x/y")
    fs.makedirs("temp://y/z")
    assert 2 == fs.create_called
    fs.makedirs("mem://x/y")
    fs.makedirs("mem://y/z")
    assert 4 == fs.create_called
    fs.writetext(os.path.join(p1, "a.txt"), "xyz")
    fs.copy(os.path.join(p1, "a.txt"), "mem://y/z/a.txt")
    assert "xyz" == fs.readtext("mem://y/z/a.txt")
    assert not fs.exists("mem://y/z/w/a.txt")
    assert 4 == fs.create_called
    fs.writetext("mem://from/a.txt", "hello")
    fs.copy("mem://from/a.txt", "mem://to/a.txt")
    assert "hello" == fs.readtext("mem://to/a.txt")
    assert 6 == fs.create_called


def test_multiple_writes(tmpdir):
    fs = FileSystem()
    path = os.path.join(tmpdir, "a.txt")
    fs.writetext(path, "1")
    fs.writetext(path, "2")
    assert "2" == fs.readtext(path)

    # auto close is important
    d2 = os.path.join(tmpdir, "x", "y")
    ff = FileSystem(auto_close=False).makedirs(d2, recreate=True)
    ff.writetext("a.txt", "3")
    ff.writetext("a.txt", "4")
    ff = FileSystem(auto_close=False).makedirs(d2, recreate=True)
    ff.writetext("a.txt", "5")
    assert "5" == ff.readtext("a.txt")


def test_glob(tmpdir):
    tmpdir = str(tmpdir)
    fs = FileSystem()
    os.makedirs(os.path.join(str(tmpdir), "d1"))
    os.makedirs(os.path.join(str(tmpdir), "d2", "d2"))
    f = open(os.path.join(str(tmpdir), "d1", "f1.txt"), "a")
    f.write("read test")
    f.close()
    f = open(os.path.join(str(tmpdir), "d2", "d2", "f2.txt"), "a")
    f.write("read test")
    f.close()
    assert {
        pfs.path.join(str(tmpdir), "d1", "f1.txt").replace("\\", "/"),
        pfs.path.join(str(tmpdir), "d2", "d2", "f2.txt").replace("\\", "/"),
    } == {x.path for x in fs.glob("**/*.txt", path=str(tmpdir))}

    fs.makedirs("mem://a/d1")
    fs.makedirs("mem://a/d2/d2")
    fs.touch("mem://a/d1/f3.txt")
    fs.touch("mem://a/d2/d2/f4.txt")
    assert {"mem://a/d1/f3.txt", "mem://a/d2/d2/f4.txt"} == {
        x.path for x in fs.glob("**/*.txt", path="mem://a")
    }


class MockFS(FileSystem):
    def __init__(self):
        super().__init__()
        self.create_called = 0

    def create_fs(self, root):
        self.create_called += 1
        return super().create_fs(root)
