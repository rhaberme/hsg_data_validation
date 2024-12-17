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

# TODO: All parameters should have defaults

# 1 Lücke
def check_gaps(df_data, value_col="value", min_border=0.0):
    pass


# 2 Konstanz
def check_constancy(df_data, delta, nr_of_sequential_measurements, value_col_name="value"):
    pass


# 3 Spanne
def check_range(df_data, lower_border = -np.inf, upper_border = np.inf, value_col_name="value"):
    df_check = pd.DataFrame(np.bitwise_or(df_data[value_col_name] < lower_border, df_data[value_col_name] > upper_border)
                            , index=df_data.index, columns=["Range Error"])
    return df_check


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


def check_for_jumps(df_data):
    pass
