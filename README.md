# Triad

[![GitHub release](https://img.shields.io/github/release/fugue-project/triad.svg)](https://GitHub.com/fugue-project/triad)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/triad.svg)](https://pypi.python.org/pypi/triad/)
[![PyPI license](https://img.shields.io/pypi/l/triad.svg)](https://pypi.python.org/pypi/triad/)
[![PyPI version](https://badge.fury.io/py/triad.svg)](https://pypi.python.org/pypi/triad/)
[![Coverage Status](https://coveralls.io/repos/github/fugue-project/triad/badge.svg)](https://coveralls.io/github/fugue-project/triad)
[![Doc](https://readthedocs.org/projects/triad/badge)](https://triad.readthedocs.org)

A collection of python utility functions for Fugue projects

## Installation
```
pip install triad
```


## Release History

### 0.3.6
* Add `transform` to Schema class

### 0.3.5
* Change pyarrow and pandas type_safe output to be consistent with pyarrow (None for pd.NaT, nan, etc)

### 0.3.4
* Add general FileSystem

### 0.3.3
* Add thread-safe cloudpicklable RunOnce class

### 0.3.2
* extracted TRIAD_DEFAULT_TIMESTAMP as a constant

### <=0.3.1
* Open sourced and docs are ready
* Added basic utility functions
* Types and schema are based on pyarrow
* A better indexed and ordered dict
* Added ParamDict
