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
import data_processing as d_p
from typing import Union


class Measurement:

    def __init__(self,
                 filepath: str,
                 name: str,
                 measurement_type: str,
                 label_value: Union[str, None],
                 label_date_time: Union[list, None],
                 is_normalized: bool = False,
                 status_available: bool = False,
                 label_status: Union[str, None] = None,
                 accepted_status: Union[str, int, None] = None,
                 validated_df: pd.DataFrame = None,
                 sep: Union[str, None] = ",",
                 measurement_number=None,
                 drop_duplicates=False,
                 are_outliers=None,
                 treshold_value=None,
                 hysteresis=None
                 ):
        self.filepath = filepath
        if is_normalized:
            self.raw_df = d_p.read_normalized_ftr(filepath, to_datetime=False, to_ns=False)
            self.status_available = True if len(self.raw_df.columns) > 1 else False
        else:
            self.raw_df = d_p.create_df_data_from_csv(filepath, date_time_col_name_s=label_date_time,
                                                      value_col_name=label_value,
                                                      ambiguous=False,
                                                      status_available=status_available,
                                                      status_col_name=label_status,
                                                      sep=sep
                                                      )
            if status_available:
                self.raw_df = d_p.replace_status_informations_with_binary(self.raw_df, allowed_status=accepted_status)
            self.status_available = status_available

        if drop_duplicates:
            self.raw_df = d_p.drop_duplicated_indices(self.raw_df)
        self.validated_df = validated_df
        self.name = name
        self.measurement_type = measurement_type
        self.year = self.recognize_year()
        self.is_normalized = is_normalized
        self.label_value = label_value
        self.label_date_time = label_date_time
        self.label_status = label_status
        self.sep = sep
        self.measurement_number = measurement_number
        self.are_outliers = are_outliers
        self.treshold_value = treshold_value
        self.hysteresis = hysteresis

    def __str__(self):
        return self.name

    def return_max_value(self):
        if self.validated_df is not None:
            return round(max(self.validated_df["value"]), 2)
        else:
            return round(max(self.raw_df["value"]), 2)

    def return_df_as_datetime(self, raw, copy=True):
        try:
            if copy:
                df = self.raw_df.copy() if raw else self.validated_df.copy()
            else:
                df = self.raw_df if raw else self.validated_df
        except AttributeError:
            return pd.DataFrame([])
        df.index = df.index.map(d_p.unixtime_to_dt64)
        return df

    def recognize_year(self):
        year = d_p.unixtime_to_dt64(self.raw_df.index.values[-120]).astype(object).year  # index = -120 as a workaround
        return year


