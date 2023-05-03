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
    author="The Fugue Development Team",
    author_email="hello@fugue.ai",
    keywords="fugue util utils utility utilities",
    url="http://github.com/fugue-project/triad",
    install_requires=[
        "numpy",
        "pandas>=1.2.0",
        "six",
        "pyarrow",
        "fs",
        "importlib-metadata; python_version < '3.8'",
    ],
    extras_require={"ciso8601": ["ciso8601"]},
    classifiers=[
        # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        "Development Status :: 5 - Production/Stable",
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
