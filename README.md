# Triad

[![GitHub release](https://img.shields.io/github/release/fugue-project/triad.svg)](https://GitHub.com/fugue-project/triad)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/triad.svg)](https://pypi.python.org/pypi/triad/)
[![PyPI license](https://img.shields.io/pypi/l/triad.svg)](https://pypi.python.org/pypi/triad/)
[![PyPI version](https://badge.fury.io/py/triad.svg)](https://pypi.python.org/pypi/triad/)
[![codecov](https://codecov.io/gh/fugue-project/triad/branch/master/graph/badge.svg?token=DGKPXDIG8M)](https://codecov.io/gh/fugue-project/triad)
[![Doc](https://readthedocs.org/projects/triad/badge)](https://triad.readthedocs.org)

[![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://join.slack.com/t/fugue-project/shared_invite/zt-jl0pcahu-KdlSOgi~fP50TZWmNxdWYQ)

A collection of python utility functions for [Fugue projects](https://github.com/fugue-project)

## Installation

```bash
pip install triad
```


## Release History

### 0.6.3

* Fix pandas warning on pd.Int64Index

### 0.6.2

* Make ciso8601 totally optional

### 0.6.1

* Support Python 3.10

### 0.6.0

* Fix extensible class bugs

### 0.5.9

* Create `extensible_class` and `extension_method` decos

### 0.5.8

* Make ciso8601 a soft dependency on windows
* Switch to codecov
* Improve documents, change to Furo theme

### 0.5.7

* Fix pandas extension data types bug

### 0.5.6

* Prepare to support [pandas extension data types](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes)
* Support Python 3.9

### 0.5.5

* Change pandas_list enforce_type df construction

### 0.5.4

* Make `FileSystem` work for windows
* Make triad fullly compatible with Windows
* Add windows tests

### 0.5.3

* Lazy evaluation for `assert_or_throw`

### 0.5.2

* For pyarrow data conversion, support np.ndarray -> list

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
