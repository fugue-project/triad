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

### 0.9.6

* Add `is_like` to Schema to compare similar schemas

### 0.9.5

* Add parse json column function to pyarrow utils
* Fix the pandas [legacy usage](https://github.com/fugue-project/triad/issues/127)

### 0.9.4

* Handle Pandas 2.2 change on [arrow string type](https://pandas.pydata.org/docs/dev/whatsnew/v2.2.0.html#other-enhancements)
* Fixed readthedocs build

### 0.9.3

* Add version constraint on fsspec

### 0.9.2

* Remove support for Python 3.7
* Add more util functions for fsspec
* Add more util functions for pyarrow
* Add sorted batch reslicers
* Add `ArrowDtype` support
* Systematically improved pandas and arrow type casts
* Remove `PandasLikeUtils.enfoce_type`
* Add more util functions for pyarrow

### 0.9.1

* Add `fsspec` as a core dependency, add io utils
* Add `py.typed`
* Improve `Schema` class comments

### 0.9.0

* Merge QPD pandas utils functions back to Triad

### 0.8.9

* Add batch reslicers
* Fix package issue (exclude tests folder)

### 0.8.8

* Add type replacement utils for pyarrow

### 0.8.7

* Fix pandas 2.0 warnings

### 0.8.6

* Fixed timestamp conversion for pandas 2.0

### 0.8.5

* Ensure pandas 2.0 compatibility
* Improve `to_schema` in `PandasUtils`

### 0.8.4

* Moved `FunctionWrapper` from Fugue into Triad
* Improved groupby apply efficiency for pandas utils

### 0.8.3

* Add `get_alter_func` to pyarrow utils

### 0.8.2

* Handle time zone in pandas_utils

### 0.8.1

* Fixed the [string column issue](https://github.com/fugue-project/fugue/issues/415)

### 0.8.0

* Support arbitrary column name

### 0.7.0

* Fixed importlib `entry_points` compatibility issue

### 0.6.9

* Remove Python 3.6 support
* Add dataframe rename utils

### 0.6.8

* Add map type support to Schema

### 0.6.7

* Parse nested fields in `expression_to_schema` util

### 0.6.6

* Improve conditional_dispatcher

### 0.6.5

* Add SerializableRLock
* Add decorator run_once

### 0.6.4

* Add function dispatcher

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
