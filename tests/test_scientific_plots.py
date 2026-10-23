import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from Webtool import scientific_plots as sp


def _series(freq="5min", periods=4000, gaps=True):
    index = pd.date_range("2023-07-20", periods=periods, freq=freq)
    t = np.arange(periods)
    hours = (index.hour + index.minute / 60).to_numpy()
    values = 20 + 3 * np.sin(2 * np.pi * hours / 24) + np.random.default_rng(0).normal(0, 0.3, periods)
    s = pd.Series(values, index=index, name="value")
    if gaps:
        s.iloc[100:150] = np.nan
    return s


def _test_csv_series():
    df = pd.read_csv("tests/test_data/data_noise.csv", parse_dates=["DateTime"]).set_index("DateTime")
    return df["Temperature [°C]"].astype(float)


def test_infer_sampling_interval():
    assert sp.infer_sampling_interval(_series()) == pd.Timedelta("5min")


def test_descriptive_statistics():
    s = _series()
    stats = sp.descriptive_statistics(s)["Value"]
    assert stats["Missing values"] == 50
    assert stats["Valid values"] == len(s) - 50
    assert stats["Sampling interval"] == "5 min"
    assert stats["Mean"] == pytest.approx(s.mean())
    assert stats["Minimum"] <= stats["Median"] <= stats["Maximum"]


def test_autocorrelation_lag_zero_is_one():
    lags, acf = sp.autocorrelation(_series(), 10)
    assert len(lags) == len(acf) == 11
    assert acf[0] == 1.0 and acf[1] > 0.5


def test_available_plots_long_series():
    keys = sp.available_plots(_series(periods=20000))
    assert {"time_series", "histogram", "diurnal", "carpet", "monthly", "acf", "psd"} <= set(keys)


def test_available_plots_short_daily_series():
    keys = sp.available_plots(_series(freq="1D", periods=10, gaps=False))
    assert "diurnal" not in keys and "carpet" not in keys and "psd" not in keys
    assert sp.available_plots(pd.Series([np.nan] * 5, index=pd.date_range("2020", periods=5))) == []


@pytest.mark.parametrize("series", [_series(periods=20000), _test_csv_series()], ids=["synthetic", "csv"])
def test_all_available_plots_render(series):
    funcs = {
        "time_series": lambda s: sp.plot_time_series(s, "T [°C]", comparison=s),
        "availability": sp.plot_data_availability,
        "histogram": lambda s: sp.plot_histogram(s, "T [°C]"),
        "qq": sp.plot_qq,
        "duration_curve": lambda s: sp.plot_duration_curve(s, "T [°C]", log_y=True),
        "diurnal": lambda s: sp.plot_diurnal_profile(s, "T [°C]"),
        "carpet": lambda s: sp.plot_carpet(s, "T [°C]"),
        "monthly": lambda s: sp.plot_monthly_boxplot(s, "T [°C]"),
        "acf": sp.plot_acf,
        "psd": lambda s: sp.plot_power_spectrum(s, "T [°C]"),
    }
    with sp.plot_style(8):
        for key in sp.available_plots(series):
            fig = funcs[key](series)
            assert isinstance(fig, Figure)
            for fmt in ("png", "pdf", "svg"):
                assert len(sp.figure_to_bytes(fig, fmt, dpi=100)) > 0


def test_figure_width_matches_setting():
    fig = sp.plot_histogram(_series(), "T", width_cm=8.5)
    assert fig.get_figwidth() == pytest.approx(8.5 / 2.54)


def test_build_zip():
    import io
    import zipfile
    data = sp.build_zip({"a.txt": b"1", "b.csv": b"2"})
    assert sorted(zipfile.ZipFile(io.BytesIO(data)).namelist()) == ["a.txt", "b.csv"]
