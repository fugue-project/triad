import os

from setuptools import find_packages, setup

from triad_version import __version__

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


def get_version() -> str:
    tag = os.environ.get("RELEASE_TAG", "")
    if "dev" in tag.split(".")[-1]:
        return tag
    if tag != "":
        assert tag == __version__, "release tag and version mismatch"
    return __version__


setup(
    name="triad",
    version=get_version(),
    packages=find_packages(include=["triad*"]),
    package_data={"triad": ["py.typed"]},
    description="A collection of python utils for Fugue projects",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    author="The Fugue Development Team",
    author_email="hello@fugue.ai",
    keywords="fugue utilities",
    url="http://github.com/fugue-project/triad",
    install_requires=[
        "numpy",
        "pandas>=1.3.5",
        "six",
        "pyarrow>=6.0.1",
        "fsspec>=2022.5.0",
        "fs",  # TODO: remove this hard dependency
    ],
    extras_require={"ciso8601": ["ciso8601"]},
    classifiers=[
        # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.8",
)
