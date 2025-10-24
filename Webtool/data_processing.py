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
import numpy as np
from datetime import datetime

def unixtime_to_dt64(ut):
    if type(ut) == pd.Timestamp:
        return ut.to_numpy()
    else:
        ut = int(ut)
        dt = datetime.utcfromtimestamp(ut)
        return np.datetime64(dt)


def df_to_datetime(df):
    df_copy = df.copy()
    df_copy.index = df_copy.index.map(unixtime_to_dt64)
    return df_copy


def create_df_data_from_csv(filepath: str,
                            date_time_col_name_s: list,
                            value_col_name: str,
                            ambiguous: bool = False,
                            status_available: bool = False,
                            status_col_name: str = None,
                            sep: str = ","):
    # TODO: Check for values to be numeric -> Include preprocessing to convert non-numeric to NaN
    # Do in separate function and call here
    # Times: Accept whatever pd.to_datetime accepts
    if len(date_time_col_name_s) == 2:
        df_data = pd.read_csv(filepath, parse_dates=[date_time_col_name_s], sep=sep)
        dt_col_name = date_time_col_name_s[0] + "_" + date_time_col_name_s[1]
    else:
        try:
            df_data = pd.read_csv(filepath, sep=sep)
        except ValueError:
            print(filepath)
            df_data = pd.read_csv(filepath, sep=";")
        dt_col_name = date_time_col_name_s[0]
        df_data[dt_col_name] = pd.to_datetime(df_data[dt_col_name], utc=True)

    df_data = df_data.rename(columns={dt_col_name: "unixtime"})
    df_data = df_data.set_index("unixtime")

    def date_to_unix(date):
        return (date - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

    df_data.index = pd.Series(df_data.index).apply(date_to_unix)

    if not status_available:
        df_data = df_data[[value_col_name]]
        df_data = df_data.rename(columns={value_col_name: "value"})

    else:
        df_data = df_data[[value_col_name, status_col_name]]
        df_data = df_data.rename(columns={value_col_name: "value", status_col_name: "status"})

    return df_data

def drop_implausible_measurements(df_values, df_inplausible, inplausible_column_name):
    df_dropped = df_values.copy()
    df_dropped.loc[df_inplausible[inplausible_column_name], 'value'] = np.nan
    return df_dropped


def drop_duplicated_indices(df_data):
    df = df_data.copy()
    return df[~df.index.duplicated(keep='first')]


def replace_status_informations_with_binary(df_data, allowed_status=None, status_col_name="status"):
    df_data_copy = df_data.copy()
    if allowed_status is None:
        allowed_status = ["Ok"]
    all_status = list(df_data_copy[status_col_name].unique())
    not_allowed_status = [str(x) for x in all_status if str(x) not in allowed_status]

    df_data_copy[status_col_name] = df_data_copy[status_col_name].replace(allowed_status, 0)
    df_data_copy[status_col_name] = df_data_copy[status_col_name].replace(not_allowed_status, 1)

    return df_data_copy




