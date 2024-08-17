import numpy as np
import pandas as pd


# 1 Lücke
def check_gaps(df_data, value_col="value", min_border=0.0):
    pass


# 2 Konstanz
def check_constancy(df_data, delta, nr_of_sequential_measurements, value_col_name="value"):
    pass


# 3 Spanne
def check_range(df_data, upper_border, lower_border, value_col_name="value"):
    pass


# 4 Ausreißer
def check_outlier(df_data, value_col_name="value", r1_perc=None, r3_perc=None):
    pass


# 5 Gradient
def check_gradient(df_data, delta=1.5, value_col_name="value"):
    pass


# 6 Rauschen
def check_noise(df_data, value_col="value"):
    return []


# 7 Drift
def check_drift(df_data, value_col="value", chunk_split=4):
    pass


def check_for_continuity(df_data):
    pass


def check_missing_days(df):
    pass


def get_min_interval(df):
    pass


def check_outliers(df, column_name='value'):
    pass


def get_height_peaks(df, column_name='value'):
    pass
