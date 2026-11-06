from typing import Any

import numpy as np
import pandas as pd

PANDAS3 = int(pd.__version__.split(".")[0]) >= 3
# default pandas dtype for pa.string() when no extension/arrow dtypes are requested
PD_STR = pd.StringDtype(na_value=np.nan) if PANDAS3 else np.dtype(str)


def assert_df_eq(
    df: pd.DataFrame,
    data: Any,
    columns: Any = None,
    digits=8,
    check_col_order: bool = False,
    check_order: bool = False,
    check_content: bool = True,
    throw=True,
) -> bool:
    df1 = df
    df2 = (
        data if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=columns)
    )
    cols = list(df1.columns)
    try:
        if not check_col_order:
            assert sorted(cols) == sorted(df2.columns), (
                f"columns mismatch {sorted(cols)}, {sorted(df2.columns)}"
            )
            df2 = df2[cols]
        else:
            assert cols == list(df2.columns), (
                f"columns mismatch {cols}, {list(df2.columns)}"
            )
        assert df1.shape == df2.shape, f"shape mismatch {df1.shape}, {df2.shape}"

        if not check_content:
            return True
        if not check_order:
            df1 = df1.sort_values(cols)
            df2 = df2.sort_values(cols)
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        pd.testing.assert_frame_equal(df1, df2, atol=10 ** (-digits), check_dtype=False)
        return True
    except AssertionError:
        if throw:
            raise
        return False
