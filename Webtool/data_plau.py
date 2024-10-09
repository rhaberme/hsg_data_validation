import numpy as np
import pandas as pd
import TSCC


# TODO how to define parameters in functions with different options

# 1 Lücke
def check_gaps(df_data, value_col="value"):
    # nans only
    # add option for missing observations etc.
    return df_data[value_col].isna().astype(float)


# 2 Konstanz
def check_constancy(df_data, nr_of_sequential_measurements, delta, value_col_name="value"):
    return TSCC.detection.BASIC_byPersistence(df_data[value_col_name],
                                              persistence_window = nr_of_sequential_measurements,
                                              delta = delta)


# 3 Spanne
def check_range(df_data, upper_border, lower_border, value_col_name="value"):
    return TSCC.detection.BASIC_byRange(df_data[value_col_name], upper=upper_border, lower=lower_border)


# 4 Ausreißer
def check_outlier(df_data, value_col_name="value", option = "IQR"):
    if option == "IQR":
        return TSCC.detection.STAT_byIQR(series=df_data[value_col_name], lo=0.25, up=0.75, k=1.5)
    elif option == "z-score":
        return TSCC.detection.STAT_byZScore(series=df_data[value_col_name], b_modified=False, z=0.5)
    else:
        return pd.Series([np.nan] * df_data.shape[0], index = df_data.index)


# 5 Gradient
def check_gradient(df_data, delta=1.5, value_col_name="value", timestep = pd.Timedelta(minutes = 30)):
    return TSCC.detection.BASIC_byStepChangeMax(df_data[value_col_name], delta, timestep)


# 6 Rauschen
def check_noise(df_data, value_col="value", option = "MAD"):
    if option == "MAD":
        # Median Absolute Deviation
        # TODO set threshold dynamically
        threshold = 5
        TSCC.detection.STAT_byDistFromCenter(df_data[value_col], threshold)
    else:
        return pd.Series([np.nan] * df_data.shape[0], index = df_data.index)


# 7 Drift
def check_drift(df_data, value_col="value", chunk_split=4, option = "movMAD"):
    if option == "movMAD":
        # TODO set threshold dynamically
        threshold = 5
        window = 10
        return TSCC.detection.STAT_byDistFromCenterRolling(df_data[value_col], threshold, window)
    else:
        return pd.Series([np.nan] * df_data.shape[0], index = df_data.index)


# 8 Sprünge
def check_for_continuity(df_data, value_col="value", option = "Binseg"):
    if option == "Binseg":
        # TODO set threshold dynamically
        n_bkps = 1
        return TSCC.detection.STAT_byBinseg(df_data[value_col], n_bkps=n_bkps)
    else:
        return pd.Series([np.nan] * df_data.shape[0], index = df_data.index)
