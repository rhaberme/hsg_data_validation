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

import numpy as np
import pandas as pd


# 1 Lücke
def check_gaps(df_data, custom_missing_values=None):
    if custom_missing_values is None:
        custom_missing_values = []
    if not isinstance(custom_missing_values, list):
        custom_missing_values = [custom_missing_values]

    missing_mask = df_data.isna() | df_data.isnull()
    for value in custom_missing_values:
        if isinstance(value, (str, int, float)):
            missing_mask = missing_mask | (df_data == value)
    df_gap = pd.DataFrame(
        {
            "Missing": missing_mask
        },
        index=df_data.index
    )

    return df_gap

# 2 Konstanz
def check_constancy(df_data, delta, nr_of_sequential_measurements, value_col_name="value"):
    pass


# 3 Spanne
def check_range(df_data, lower_border=-np.inf, upper_border=np.inf, value_col_name="value"):
    df_check = pd.DataFrame(
        np.bitwise_or(df_data[value_col_name] < lower_border, df_data[value_col_name] > upper_border)
        , index=df_data.index, columns=["Range Error"])
    return df_check


# 4 Ausreißer
def check_outlier(df_data, value_col_name="value", method="std_method", threshold_multiplier=3, iqr_multiplier=1.5):
    if value_col_name not in df_data.columns:
        raise ValueError(f"Column '{value_col_name}' not in dataframe")

    if method == "std_method":
        assert threshold_multiplier
        mean_value = df_data[value_col_name].mean()
        std_dev = df_data[value_col_name].std()

        upper_limit = mean_value + threshold_multiplier * std_dev
        lower_limit = mean_value - threshold_multiplier * std_dev

        df_outliers = pd.DataFrame(
            {
                "Outlier": (df_data[value_col_name] > upper_limit) | (df_data[value_col_name] < lower_limit)
            },
            index=df_data.index
        )
    elif method == "iqr_method":
        q1 = df_data[value_col_name].quantile(0.25)
        q3 = df_data[value_col_name].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        df_outliers = pd.DataFrame(
            {
                "Outlier": (df_data[value_col_name] < lower_bound) | (df_data[value_col_name] > upper_bound)
            },
            index=df_data.index
        )
    else:
        raise ValueError(f"{method} is not one of the defined outlier detection methods")

    return df_outliers


# 5 Gradient
def check_gradient(df_data, delta=1.5, value_col_name="value"):
    pass

def check_gradients(df_data, value_col_name="value", gradient_threshold=None):
    if value_col_name not in df_data.columns:
        raise ValueError(f"Column '{value_col_name}' not in dataframe")

    if gradient_threshold is None:
        raise ValueError("Gradient_threshold not defined")

    gradients = df_data[value_col_name].diff().abs()
    df_high_gradients = pd.DataFrame(
        {
            "High_Gradient": gradients > gradient_threshold
        },
        index=df_data.index
    )

    # first row will have nan for gradient --> set it to False
    df_high_gradients.loc[df_high_gradients.index[0], 'High_Gradient'] = False

    return df_high_gradients


# 6 Rauschen
def check_noise(df_data, value_col="value"):
    return []


# 7 Drift
def check_drift(df_data, value_col="value", chunk_split=4):
    pass


def check_for_jumps(df_data):
    pass
