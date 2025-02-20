
# TODO: List with all functions and test them automatically:
# TODO: Do they provide correct output format for different inputs?



# TODO: Test all plausibility checks individually for correct outputs (with different parameters etc.)
import pytest
import pandas as pd
from Webtool.data_plau import *

def test_jumps():
    df_data = pd.read_csv("tests/test_data/data_shift.csv")
    value_col_name = "Temperature [°C]"
    isjump = check_for_jumps(df_data, window=2, threshold=1, value_col_name=value_col_name)
    true_correct = all(isjump["Jump"].iloc[[*range(39, 43), *range(58, 62), *range(104, 108)]])
    false_correct = not any(isjump["Jump"].iloc[[*range(0, 39), *range(43, 58), *range(62, 104), *range(108, 126)]])
    assert true_correct and false_correct