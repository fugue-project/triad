# Triad

[![GitHub release](https://img.shields.io/github/release/fugue-project/triad.svg)](https://GitHub.com/fugue-project/triad)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/triad.svg)](https://pypi.python.org/pypi/triad/)
[![PyPI license](https://img.shields.io/pypi/l/triad.svg)](https://pypi.python.org/pypi/triad/)
[![PyPI version](https://badge.fury.io/py/triad.svg)](https://pypi.python.org/pypi/triad/)
[![Coverage Status](https://coveralls.io/repos/github/fugue-project/triad/badge.svg)](https://coveralls.io/github/fugue-project/triad)
[![Doc](https://readthedocs.org/projects/triad/badge)](https://triad.readthedocs.org)

[Join Fugue-Project on Slack](https://join.slack.com/t/fugue-project/shared_invite/zt-he6tcazr-OCkj2GEv~J9UYoZT3FPM4g)

A collection of python utility functions for [Fugue projects](https://github.com/fugue-project)

## Installation
```
pip install triad
```


## Release History

### 0.5.1

* Update get_caller_global_local_vars to access any stack

### 0.5.0

* Fix to_type on full type path

### 0.4.9

* Fix numpy warning

### 0.4.6

* Improve pandas like utils `enforce` method to handle str -> bool

### 0.4.5

* Fixed pandas -> arrow datetime conversion issue

### 0.4.4

* Improved FileSystem compatibility with Windows
* Add overwrite expression for Schema class
* Fixed github actions

### 0.4.3

* Refactored `str_to_type`, `str_to_instance` and `to_function` to use `eval`

### 0.4.2

* Fix a bug in pandas like safe_groupby_apply

### 0.4.1

* Improvement on group by apply
* Improvement on environment setup

### 0.4.0

* Prepare for Fugue open source

### 0.3.8

* Change to Apache 2.0 license

### 0.3.7

* Add pyarrow binary type support

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
