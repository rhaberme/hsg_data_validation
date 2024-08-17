import pandas as pd
import numpy as np
from datetime import datetime


def dt64_to_unixtime(dt64, timezone="GMT+2"):
    if timezone == "GMT+2":
        return dt64.astype('datetime64[s]').astype('int') + 7200
    else:
        return dt64.astype('datetime64[s]').astype('int')


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

    if len(date_time_col_name_s) == 2:
        df_data = pd.read_csv(filepath, parse_dates=[date_time_col_name_s], sep=sep)
        dt_col_name = date_time_col_name_s[0] + "_" + date_time_col_name_s[1]
    else:
        try:
            df_data = pd.read_csv(filepath, parse_dates=[date_time_col_name_s[0]], sep=sep)
        except ValueError:
            print(filepath)
            df_data = pd.read_csv(filepath, parse_dates=[date_time_col_name_s[0]], sep=";")
        dt_col_name = date_time_col_name_s[0]

    df_data = df_data.rename(columns={dt_col_name: "unixtime"})
    df_data = df_data.set_index("unixtime")
    try:
        df_data = df_data.tz_localize('Europe/Berlin', ambiguous=ambiguous, nonexistent='shift_forward').tz_convert('UTC')
    except TypeError:
        pass

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


def normalize_csv(filepath: str,
                  output_path: str,
                  date_time_col_name_s: list,
                  value_col_name: str,
                  ambiguous: bool = False,
                  status_available: bool = False,
                  status_col_name: str = None,
                  allowed_status: list = None,
                  sep: str = ",",
                  to_ns: bool = False):
    if len(date_time_col_name_s) == 2:
        df_data = pd.read_csv(filepath, parse_dates=[date_time_col_name_s], sep=sep)
        dt_col_name = date_time_col_name_s[0] + "_" + date_time_col_name_s[1]
    else:
        df_data = pd.read_csv(filepath, parse_dates=[date_time_col_name_s[0]], sep=sep)
        dt_col_name = date_time_col_name_s[0]

    df_data = df_data.set_index(dt_col_name)
    df_data = df_data.tz_localize('Europe/Berlin', ambiguous=ambiguous, nonexistent='shift_forward').tz_convert('UTC')

    def date_to_unix(date):
        return (date - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')

    df_data.index = pd.Series(df_data.index).apply(date_to_unix)

    if not status_available:
        df_data = df_data[[value_col_name]]
        df_data = df_data.rename(columns={value_col_name: "value"})

    else:
        df_data = df_data[[value_col_name, status_col_name]]
        df_data = df_data.rename(columns={value_col_name: "value", status_col_name: "status"})

    print(f'CSV-file normalized and saved as {output_path}')
    df_data = df_data.reset_index()
    df_data = df_data.rename(columns={dt_col_name: 'unixtime'})

    if status_available:
        df_data = replace_status_informations_with_binary(df_data, allowed_status=allowed_status,
                                                          status_col_name="status")
    df_data = df_data.sort_index()
    df_data.to_feather(output_path)


def read_normalized_ftr(filepath: str, to_datetime=False, to_ns=False):
    df = pd.read_feather(filepath)
    df.set_index("unixtime", inplace=True)
    if to_ns and not to_datetime:
        try:
            df.index = pd.to_datetime(df.index, unit='s')  # delete as soon as every *.ftr-file is normalized
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            pass
        df.index = df.index.astype(np.int64)  # delete as soon as every *.ftr-file is normalized
    df.sort_index(inplace=True)  # delete as soon as every *.ftr-file is normalized
    if to_datetime:
        df.index = df.index.map(unixtime_to_dt64)
    return df


def drop_implausible_measurements(df_data, implausible_dates, value_col_name="value"):
    df_data_copy = df_data.copy()
    for date in implausible_dates:
        try:
            df_data_copy.loc[df_data_copy.index == date, value_col_name] = np.nan
        except IndexError:
            print("Reihe nicht gefunden")
            pass
    return df_data_copy


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




