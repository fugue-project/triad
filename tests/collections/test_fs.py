import os
from os.path import exists

from pytest import raises
from triad.collections.fs import FileSystem, FSPath


def test_fspath():
    p = FSPath("/a//b.txt")
    assert "" == p.scheme
    assert "/" == p.root
    assert "a/b.txt" == p.relative_path

    p = FSPath("//a.txt")
    assert "" == p.scheme
    assert "/" == p.root
    assert "a.txt" == p.relative_path

    p = FSPath("file://a/b.txt")
    assert "" == p.scheme
    assert "/" == p.root
    assert "a/b.txt" == p.relative_path

    p = FSPath("hdfs://a/b.txt")
    assert "hdfs" == p.scheme
    assert "hdfs://a" == p.root
    assert "b.txt" == p.relative_path

    p = FSPath("temp://a/b")
    assert "temp" == p.scheme
    assert "temp://a" == p.root
    assert "b" == p.relative_path

    p = FSPath("/temp:/a/b")
    assert "temp" == p.scheme
    assert "temp://a" == p.root
    assert "b" == p.relative_path

    raises(ValueError, lambda: FSPath(None))
    raises(ValueError, lambda: FSPath(""))
    raises(ValueError, lambda: FSPath("a.txt"))
    raises(ValueError, lambda: FSPath("hdfs://"))


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


class MockFS(FileSystem):
    def __init__(self):
        super().__init__()
        self.create_called = 0

    def create_fs(self, root):
        self.create_called += 1
        return super().create_fs(root)
