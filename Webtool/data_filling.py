# hsg_data_validation: Funktionen für die Datenvalidierung
# Copyright (C) 2024 HSGSim Arbeitsgruppe "Messdaten und Machine Learning"
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd

def fill_nan_interp(df_data, value_col_name="value"):
    df_data_copy = df_data.copy()
    df_data_copy[value_col_name].interpolate(inplace=True)
    return df_data_copy


def fill_nan_mean(df_data, value_col_name="value"):
    df_data_copy = df_data.copy()
    df_data_copy[value_col_name] = df_data_copy[value_col_name].fillna(df_data_copy[value_col_name].mean())
    return df_data_copy


def fill_nan_null(df_data, value_col_name="value"):
    df_data_copy = df_data.copy()
    df_data_copy[value_col_name] = df_data_copy[value_col_name].fillna(0)
    return df_data_copy


def fill_nan_rollmean(df_data, value_col_name="value"):
    df_data_copy = df_data.copy()
    df_interp = fill_nan_interp(df_data_copy, value_col_name=value_col_name)

    df_data_copy[value_col_name] = df_data_copy[value_col_name].fillna(df_interp[value_col_name].rolling(window=10).mean())
    return df_data_copy

"""
# import TSCC

def fill_nan_interp(df_data, value_col_name="value"):
    df_data["isError"] = df_data[value_col_name].isna()
    config = TSCC.preprocessing.Config(colname_raw = value_col_name)
    return TSCC.correction.STAT_byInterpolation(df_fea, None, "isError", config)


def fill_nan_mean(df_data, value_col_name="value"):
    config = TSCC.preprocessing.Config(colname_raw = value_col_name)
    return TSCC.correction.BASE_NA_byMode(df_data, None, config)


def fill_nan_null(df_data, value_col_name="value"):
    df_data["isError"] = df_data[value_col_name].isna()
    config = TSCC.preprocessing.Config(colname_raw = value_col_name)
    return TSCC.correction.STAT_byFilling(df_fea, None, "isError", config, filler = "ffill")


def fill_nan_rollingmean(df_data, value_col_name="value"):
    df_data["isError"] = df_data[value_col_name].isna()
    config = TSCC.preprocessing.Config(colname_raw = value_col_name)
    return TSCC.correction.STAT_byRollingMean(df_fea, None, "isError", config)

"""