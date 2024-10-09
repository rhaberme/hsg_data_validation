import pandas as pd
import TSCC


def fill_nan_interp(df_data, value_col_name="value"):
    df_data["isError"] = df_data[value_col_name].isna()
    config = TSCC.preprocessing.Config(colname_raw = value_col_name)
    return TSCC.correction.STAT_byInterpolation(df_data, None, "isError", config)


def fill_nan_mean(df_data, value_col_name="value"):
    config = TSCC.preprocessing.Config(colname_raw = value_col_name)
    return TSCC.correction.BASE_NA_byMode(df_data, None, config)


def fill_nan_null(df_data, value_col_name="value"):
    df_data["isError"] = df_data[value_col_name].isna()
    config = TSCC.preprocessing.Config(colname_raw = value_col_name)
    return TSCC.correction.STAT_byFilling(df_data, None, "isError", config, filler = "ffill")


def fill_nan_rollingmean(df_data, value_col_name="value"):
    df_data["isError"] = df_data[value_col_name].isna()
    config = TSCC.preprocessing.Config(colname_raw = value_col_name)
    return TSCC.correction.STAT_byRollingMean(df_data, None, "isError", config)

