import os
from os.path import exists

from pytest import raises
from triad.collections.fs import FileSystem, _FSPath


def test__FSPath():
    p = _FSPath("/a//b.txt")
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

    # Windows test cases
    p = _FSPath("c:\\folder\\myfile.txt")
    assert "" == p.scheme
    assert "c:/" == p.root
    assert "folder/myfile.txt" == p.relative_path

    p = _FSPath("\\\\tmp\\tmp.txt")
    assert "" == p.scheme
    assert "/" == p.root
    assert "tmp/tmp.txt" == p.relative_path

    p = _FSPath("\\\\123.123.123.123\\share\\folder\\myfile.txt")
    assert "" == p.scheme
    assert "/" == p.root
    assert "123.123.123.123/share/folder/myfile.txt" == p.relative_path

    raises(ValueError, lambda: _FSPath(None))
    raises(ValueError, lambda: _FSPath(""))
    raises(ValueError, lambda: _FSPath("a.txt"))
    raises(ValueError, lambda: _FSPath("hdfs://"))


def test_fs(tmpdir):
    # Tests to read and write with tmpdir without FS
    tmpfile = os.path.join(tmpdir, "f.txt")
    f = open(tmpfile, "a")
    f.write("read test")
    f.close()
    f = open(tmpfile, "r")
    assert f.read() == "read test"

    p1 = os.path.join(tmpdir, "a")
    p2 = os.path.join(tmpdir, "b")
    assert not os.path.exists(p1)
    assert not os.path.exists(p2)
    fs = MockFS()
    fs.makedirs(p1)
    fs.makedirs(p2)
    assert os.path.exists(p1) and os.path.isdir(p1)
    assert os.path.exists(p2) and os.path.isdir(p2)
    assert 1 == fs.create_called
    fs.makedirs("temp://x/y")
    fs.makedirs("temp://y/z")
    assert 3 == fs.create_called
    fs.makedirs("mem://x/y")
    fs.makedirs("mem://y/z")
    assert 5 == fs.create_called
    fs.writetext(os.path.join(p1, "a.txt"), "xyz")
    fs.copy(os.path.join(p1, "a.txt"), "mem://y/z/a.txt")
    assert "xyz" == fs.readtext("mem://y/z/a.txt")
    assert not fs.exists("mem://y/z/w/a.txt")
    assert 5 == fs.create_called
    fs.writetext("mem://from/a.txt", "hello")
    fs.copy("mem://from/a.txt", "mem://to/a.txt")
    assert "hello" == fs.readtext("mem://to/a.txt")
    assert 7 == fs.create_called


def test_glob(tmpdir):
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
        os.path.join(str(tmpdir), "d1", "f1.txt"),
        os.path.join(str(tmpdir), "d2", "d2", "f2.txt"),
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
