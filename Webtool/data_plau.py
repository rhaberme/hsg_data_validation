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
    custom_missing_values = None if custom_missing_values == "" else custom_missing_values
    if custom_missing_values is None:
        custom_missing_values = []
    if not isinstance(custom_missing_values, list):
        custom_missing_values = [custom_missing_values]

    missing_mask = df_data.isna() | df_data.isnull()

    for value in custom_missing_values:
        if isinstance(value, (str, int, float)):
            missing_mask |= (df_data == value)

    df_gap = pd.DataFrame({
        "Missing": missing_mask.any(axis=1)
    }, index=df_data.index)

    return df_gap



# 2 Konstanz
def check_constancy(df_data, window=2, threshold=1e-8, min_std=0, value_col_name="value", method="threshold"):
    if value_col_name not in df_data.columns:
        raise ValueError(f"Column '{value_col_name}' not in dataframe")

    if method == "threshold": # If maximum difference is too low in the window
        # Compute rolling difference
        roll = df_data[value_col_name].rolling(window=window, center=True)
        delta = roll.max() - roll.min()
        # Check if it exceeds the minimum
        df_constant = pd.DataFrame(
            {
                "Constancy": (delta < threshold)
            },
            index=df_data.index
        )
        # In the beginning, window is outside and delta is nan -> treat as non-constant
        df_constant.loc[df_constant.index[range(window-1)], 'Constancy'] = False
    elif method == "std": # If std is too low in the window
        # Compute rolling std
        std = df_data[value_col_name].rolling(window=window, center=True).std()
        # Check if it exceeds the minimum
        df_constant = pd.DataFrame(
            {
                "Constancy": (std < min_std)
            },
            index=df_data.index
        )
        # In the beginning, window is outside and std is nan -> treat as non-constant
        df_constant.loc[df_constant.index[range(window-1)], 'Constancy'] = False
    else:
        raise ValueError(f"{method} is not one of the defined constancy detection methods")

    return df_constant


# 3 Spanne
def check_range(df_data, lower_border=-np.inf, upper_border=np.inf, value_col_name="value"):
    df_check = pd.DataFrame(
        np.bitwise_or(df_data[value_col_name] < lower_border, df_data[value_col_name] > upper_border)
        , index=df_data.index, columns=["Range Error"])
    return df_check


# 4 Ausreißer
def check_outlier(df_data, value_col_name="value", method="std_method", std_multiplier=3, iqr_multiplier=1.5):
    if value_col_name not in df_data.columns:
        raise ValueError(f"Column '{value_col_name}' not in dataframe")

    if method == "std_method":
        assert std_multiplier
        mean_value = df_data[value_col_name].mean()
        std_dev = df_data[value_col_name].std()

        upper_limit = mean_value + std_multiplier * std_dev
        lower_limit = mean_value - std_multiplier * std_dev

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
def check_noise(df_data, window_size, threshold, value_col="value"):
    
    rolling_std = df_data.rolling(window=window_size, min_periods=1).std()
    df_noise = df_data.where(rolling_std <= threshold, np.nan)
    
    return df_noise


# 7 Drift
def check_drift(df_data, window=10, threshold=0.1, zero=0,  method="mean", value_col_name="value"):
    if value_col_name not in df_data.columns:
        raise ValueError(f"Column '{value_col_name}' not in dataframe")

    if method == "mean": # If maximum difference is too low in the window
        # Compute rolling mean of the difference
        rollmean = df_data[value_col_name].diff().rolling(window=window, center=True).mean()
        # Check if it exceeds the minimum
        df_drift = pd.DataFrame(
            {
                "Drift": (rollmean > threshold)
            },
            index=df_data.index
        )
        # In the beginning, window is outside and mean is nan -> treat as non-drifting
        df_drift.loc[df_drift.index[range(window-1)], 'Drift'] = False
    elif method == "zero": # If std is too low in the window
        # Compute rolling average
        rollmean = df_data[value_col_name].rolling(window=window, center=True).mean()
        # Check if it exceeds the minimum
        df_drift = pd.DataFrame(
            {
                "Drift": (abs(rollmean - zero) > threshold)
            },
            index=df_data.index
        )
        # In the beginning, window is outside and mean is nan -> treat as non-drifting
        df_drift.loc[df_drift.index[range(window-1)], 'Drift'] = False
    else:
        raise ValueError(f"{method} is not one of the defined constancy detection methods")

    return df_drift


def check_for_jumps(df_data, window=2, threshold=1, value_col_name="value"):
    if value_col_name not in df_data.columns:
        raise ValueError(f"Column '{value_col_name}' not in dataframe")

    # Compute rolling mean left and right
    rollmean_left = df_data[value_col_name].rolling(window=window, center=False, closed='left').mean()
    rollmean_right = df_data[value_col_name].iloc[::-1].rolling(window=window, center=False, closed='left').mean().iloc[::-1]
    # Check if it exceeds the minimum
    df_jump = pd.DataFrame(
        {
            "Jump": (abs(rollmean_left - rollmean_right) > threshold)
        },
        index=df_data.index
    )
    # In the beginning and the ending, window is outside and mean is nan -> treat as non-drifting
    df_jump.loc[df_jump.index[range(window - 1)], 'Jump'] = False
    df_jump.loc[df_jump.index[range(-1, -window, -1)], 'Jump'] = False

    return df_jump
