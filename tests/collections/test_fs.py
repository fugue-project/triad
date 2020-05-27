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

    raises(ValueError, lambda: _FSPath(None))
    raises(ValueError, lambda: _FSPath(""))
    raises(ValueError, lambda: _FSPath("a.txt"))
    raises(ValueError, lambda: _FSPath("hdfs://"))


def test_fs(tmpdir):
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
    

class MockFS(FileSystem):
    def __init__(self):
        super().__init__()
        self.create_called = 0

    def create_fs(self, root):
        self.create_called += 1
        return super().create_fs(root)
