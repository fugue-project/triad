from setuptools import setup, find_packages
from triad_version import __version__


with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="triad",
    version=__version__,
    packages=find_packages(),
    description="A collection of python utils for Fugue projects",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    author="Han Wang",
    author_email="goodwanghan@gmail.com",
    keywords="fugue util utils utility utilities",
    url="http://github.com/fugue-project/triad",
    install_requires=[
        "pandas",
        "six",
        "pyarrow",
        "fs",
        "importlib-metadata; python_version < '3.8'",
    ],
    extras_require={"ciso8601": ["ciso8601"]},
    classifiers=[
        # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.7",
)
