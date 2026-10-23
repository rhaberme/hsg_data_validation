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

"""Publication-quality figures and descriptive statistics for a single time series.

All functions take a ``pd.Series`` with a ``DatetimeIndex`` (NaN = missing value)
and return a ``matplotlib.figure.Figure``. The pyplot state machine is not used,
so the functions are safe to call from Streamlit's worker threads.
"""

import io
import zipfile
from contextlib import contextmanager

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import signal, stats

CM = 1 / 2.54  # inch per cm

# Okabe-Ito palette: colour-blind safe and readable in greyscale print
COLORS = {
    "main": "#0072B2",
    "second": "#D55E00",
    "accent": "#009E73",
    "grey": "#7F7F7F",
    "light": "#BFD7EA",
}

# Common journal column widths in cm
FIGURE_WIDTHS_CM = {
    "Single column (8.5 cm)": 8.5,
    "1.5 column (12 cm)": 12.0,
    "Double column (17.5 cm)": 17.5,
}


def rc_params(font_size=8, font_family="sans-serif"):
    return {
        "font.size": font_size,
        "font.family": font_family,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,  # embed TrueType fonts -> editable text in PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep text as text in SVG
    }


@contextmanager
def plot_style(font_size=8, font_family="sans-serif"):
    with matplotlib.rc_context(rc_params(font_size, font_family)):
        yield


def _new_figure(width_cm, aspect=0.62):
    return Figure(figsize=(width_cm * CM, width_cm * CM * aspect), layout="constrained")


def _clean(series):
    return series.dropna().astype(float)


def infer_sampling_interval(series):
    """Median time step of the index as ``pd.Timedelta`` (None if undeterminable)."""
    if len(series.index) < 2:
        return None
    diffs = np.diff(series.index.values).astype("timedelta64[ns]")
    return pd.Timedelta(np.median(diffs))


def _format_timedelta(td):
    if td is None:
        return "n/a"
    seconds = td.total_seconds()
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= size and seconds % size == 0:
            return f"{int(seconds // size)} {unit}"
    return f"{seconds:g} s"


def _date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def descriptive_statistics(series):
    """Descriptive statistics as a one-column DataFrame (index = statistic)."""
    values = _clean(series)
    n_total = len(series)
    n_valid = len(values)
    dt = infer_sampling_interval(series)
    mean = values.mean() if n_valid else np.nan
    std = values.std(ddof=1) if n_valid > 1 else np.nan
    result = {
        "Start": series.index.min(),
        "End": series.index.max(),
        "Sampling interval": _format_timedelta(dt),
        "Number of time steps": n_total,
        "Valid values": n_valid,
        "Missing values": n_total - n_valid,
        "Completeness [%]": 100 * n_valid / n_total if n_total else np.nan,
        "Mean": mean,
        "Standard deviation": std,
        "Coefficient of variation [-]": std / mean if mean else np.nan,
        "Minimum": values.min() if n_valid else np.nan,
        "5th percentile": values.quantile(0.05) if n_valid else np.nan,
        "25th percentile": values.quantile(0.25) if n_valid else np.nan,
        "Median": values.median() if n_valid else np.nan,
        "75th percentile": values.quantile(0.75) if n_valid else np.nan,
        "95th percentile": values.quantile(0.95) if n_valid else np.nan,
        "Maximum": values.max() if n_valid else np.nan,
        "Skewness [-]": stats.skew(values, bias=False) if n_valid > 2 else np.nan,
        "Excess kurtosis [-]": stats.kurtosis(values, bias=False) if n_valid > 3 else np.nan,
        "Lag-1 autocorrelation [-]": series.astype(float).autocorr(1) if n_valid > 2 else np.nan,
    }
    return pd.DataFrame({"Value": pd.Series(result, dtype=object)})


def autocorrelation(series, max_lag):
    """Autocorrelation for lags 0..max_lag, using pairwise-complete observations."""
    values = series.astype(float)
    lags = np.arange(0, max_lag + 1)
    acf = np.array([1.0] + [values.autocorr(int(k)) for k in lags[1:]])
    return lags, acf


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_time_series(series, ylabel, width_cm=17.5, comparison=None, comparison_label="Raw",
                     series_label="Time series", mark_gaps=True):
    fig = _new_figure(width_cm, aspect=0.38)
    ax = fig.add_subplot()
    if comparison is not None:
        ax.plot(comparison.index, comparison.values, color=COLORS["grey"], lw=0.6,
                alpha=0.8, label=comparison_label)
    ax.plot(series.index, series.values, color=COLORS["main"], label=series_label)
    if mark_gaps:
        missing = series.index[series.isna()]
        if len(missing):
            ax.plot(missing, np.zeros(len(missing)), "|", color=COLORS["second"], ms=4, mew=0.6,
                    transform=ax.get_xaxis_transform(), clip_on=False, label="Missing value")
    ax.set_ylabel(ylabel)
    _date_axis(ax)
    ax.margins(x=0)
    if comparison is not None or (mark_gaps and series.isna().any()):
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), ncols=3)
    return fig


def plot_histogram(series, ylabel, width_cm=8.5, bins="fd"):
    values = _clean(series)
    fig = _new_figure(width_cm, aspect=0.75)
    ax = fig.add_subplot()
    if isinstance(bins, str):
        bins = min(len(np.histogram_bin_edges(values, bins=bins)) - 1, 100)
    ax.hist(values, bins=bins, density=True, color=COLORS["light"], edgecolor=COLORS["main"],
            linewidth=0.4, label="Histogram")
    grid = np.linspace(values.min(), values.max(), 300)
    if values.nunique() > 1:
        sample = values.sample(min(len(values), 20000), random_state=0)
        ax.plot(grid, stats.gaussian_kde(sample)(grid), color=COLORS["main"], label="KDE")
        ax.plot(grid, stats.norm.pdf(grid, values.mean(), values.std()), "--",
                color=COLORS["second"], label="Normal fit")
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Probability density")
    ax.legend(loc="upper right")
    text = (f"n = {len(values):,}\nmean = {values.mean():.3g}\nSD = {values.std():.3g}\n"
            f"skew = {stats.skew(values, bias=False):.2f}")
    ax.text(0.02, 0.97, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=matplotlib.rcParams["legend.fontsize"])
    return fig


def plot_qq(series, width_cm=8.5):
    values = _clean(series)
    if len(values) > 20000:
        values = values.sample(20000, random_state=0)
    (theoretical, ordered), (slope, intercept, r) = stats.probplot(values, dist="norm")
    fig = _new_figure(width_cm, aspect=0.85)
    ax = fig.add_subplot()
    ax.plot(theoretical, ordered, "o", ms=2, mfc="none", mec=COLORS["main"], mew=0.5,
            label="Observations")
    ax.plot(theoretical, slope * theoretical + intercept, color=COLORS["second"],
            label=f"Reference line ($R^2$ = {r ** 2:.3f})")
    ax.set_xlabel("Theoretical quantiles (standard normal)")
    ax.set_ylabel("Sample quantiles")
    ax.legend(loc="upper left")
    return fig


def plot_duration_curve(series, ylabel, width_cm=8.5, log_y=False):
    values = np.sort(_clean(series).values)[::-1]
    exceedance = 100 * np.arange(1, len(values) + 1) / (len(values) + 1)  # Weibull plotting position
    fig = _new_figure(width_cm, aspect=0.75)
    ax = fig.add_subplot()
    ax.plot(exceedance, values, color=COLORS["main"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exceedance probability [%]")
    ax.set_ylabel(ylabel)
    if log_y and values.min() > 0:
        ax.set_yscale("log")
    return fig


def plot_diurnal_profile(series, ylabel, width_cm=8.5):
    values = _clean(series)
    hour = values.index.hour + values.index.minute / 60
    grouped = values.groupby(np.floor(hour).astype(int))
    q = grouped.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    fig = _new_figure(width_cm, aspect=0.75)
    ax = fig.add_subplot()
    ax.fill_between(q.index, q[0.05], q[0.95], color=COLORS["light"], alpha=0.6, lw=0,
                    label="5–95 % range")
    ax.fill_between(q.index, q[0.25], q[0.75], color=COLORS["main"], alpha=0.35, lw=0,
                    label="Interquartile range")
    ax.plot(q.index, q[0.5], color=COLORS["main"], marker="o", ms=2, label="Median")
    ax.plot(q.index, grouped.mean(), "--", color=COLORS["second"], label="Mean")
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    return fig


def plot_carpet(series, ylabel, width_cm=17.5, cmap="viridis"):
    """Carpet plot (heat map): day on the x axis, hour of day on the y axis, hourly mean as colour."""
    values = _clean(series)
    hourly = values.resample("1h").mean()
    frame = pd.DataFrame({"value": hourly.values,
                          "date": hourly.index.normalize(),
                          "hour": hourly.index.hour})
    pivot = frame.pivot_table(index="hour", columns="date", values="value", dropna=False)
    pivot = pivot.reindex(range(24))
    dates = mdates.date2num(pivot.columns.to_pydatetime())
    fig = _new_figure(width_cm, aspect=0.38)
    ax = fig.add_subplot()
    mesh = ax.pcolormesh(np.append(dates, dates[-1] + 1), np.arange(25),
                         np.ma.masked_invalid(pivot.values), cmap=cmap, shading="flat",
                         rasterized=True)
    ax.grid(False)
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 6))
    ax.set_ylabel("Hour of day")
    _date_axis(ax)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01, aspect=25)
    cbar.set_label(ylabel)
    cbar.outline.set_linewidth(0.4)
    return fig


def plot_monthly_boxplot(series, ylabel, width_cm=17.5):
    values = _clean(series)
    periods = values.index.to_period("M")
    groups = [(str(p), g.values) for p, g in values.groupby(periods)]
    fig = _new_figure(width_cm, aspect=0.4)
    ax = fig.add_subplot()
    ax.boxplot([g for _, g in groups], tick_labels=[label for label, _ in groups], whis=(5, 95),
               showfliers=False, widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=COLORS["light"], edgecolor=COLORS["main"], lw=0.6),
               medianprops=dict(color=COLORS["second"], lw=1),
               whiskerprops=dict(color=COLORS["main"], lw=0.6),
               capprops=dict(color=COLORS["main"], lw=0.6))
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    if len(groups) > 12:
        ax.tick_params(axis="x", labelrotation=90)
    return fig


def plot_acf(series, width_cm=8.5, max_lag=None):
    dt = infer_sampling_interval(series)
    n_valid = series.notna().sum()
    if max_lag is None:
        # up to two days, but at most 500 lags and a quarter of the series length
        max_lag = int(pd.Timedelta("2D") / dt) if dt is not None and dt > pd.Timedelta(0) else 50
        max_lag = max(1, min(max_lag, 500, len(series) // 4))
    lags, acf = autocorrelation(series, max_lag)
    lag_hours = lags * dt.total_seconds() / 3600
    fig = _new_figure(width_cm, aspect=0.75)
    ax = fig.add_subplot()
    conf = 1.96 / np.sqrt(n_valid)
    ax.axhspan(-conf, conf, color=COLORS["light"], alpha=0.6, lw=0, label="95 % CI (white noise)")
    ax.axhline(0, color="black", lw=0.5)
    if len(lags) <= 60:
        ax.vlines(lag_hours, 0, acf, color=COLORS["main"], lw=0.8)
        ax.plot(lag_hours, acf, "o", ms=2, color=COLORS["main"], label="ACF")
    else:
        ax.plot(lag_hours, acf, color=COLORS["main"], label="ACF")
    ax.set_xlim(0, lag_hours[-1])
    ax.set_ylim(min(-0.2, np.nanmin(acf) - 0.05), 1.05)
    ax.set_xlabel("Lag [h]")
    ax.set_ylabel("Autocorrelation [-]")
    ax.legend(loc="best")
    return fig


def plot_power_spectrum(series, ylabel, width_cm=8.5):
    """Welch power spectral density. Gaps are linearly interpolated, leading/trailing NaN removed."""
    dt = infer_sampling_interval(series)
    values = series.astype(float).interpolate(method="linear", limit_area="inside").dropna()
    fs = 86400 / dt.total_seconds()  # samples per day
    freq, psd = signal.welch(values.values - values.mean(), fs=fs, nperseg=min(len(values), 4096),
                             detrend="linear")
    freq, psd = freq[1:], psd[1:]
    fig = _new_figure(width_cm, aspect=0.75)
    ax = fig.add_subplot()
    ax.loglog(freq, psd, color=COLORS["main"])
    for f, label in ((1, "1 d"), (2, "12 h"), (1 / 7, "7 d")):
        if freq.min() <= f <= freq.max():
            ax.axvline(f, color=COLORS["grey"], ls=":", lw=0.6)
            ax.text(f, 0.98, label, transform=ax.get_xaxis_transform(), rotation=90,
                    ha="right", va="top", color=COLORS["grey"],
                    fontsize=matplotlib.rcParams["legend.fontsize"])
    ax.set_xlabel("Frequency [1/d]")
    ax.set_ylabel(f"PSD [({ylabel})² · d]")
    return fig


def plot_data_availability(series, width_cm=17.5, freq=None):
    span = series.index.max() - series.index.min()
    if freq is None:
        freq = "1D" if span <= pd.Timedelta("400D") else "MS"
    completeness = 100 * series.notna().resample(freq).mean()
    fig = _new_figure(width_cm, aspect=0.25)
    ax = fig.add_subplot()
    width = 0.9 if freq == "1D" else 25
    colors = [COLORS["main"] if c >= 90 else COLORS["second"] for c in completeness.fillna(0)]
    ax.bar(completeness.index, completeness.values, width=width, color=colors, align="edge", lw=0)
    ax.axhline(90, color=COLORS["grey"], ls="--", lw=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Completeness [%]")
    ax.grid(axis="x", visible=False)
    _date_axis(ax)
    ax.margins(x=0.005)
    return fig


# ---------------------------------------------------------------------------
# Selection and export
# ---------------------------------------------------------------------------

def available_plots(series):
    """Return the keys of all plots that are meaningful for the given series."""
    values = _clean(series)
    dt = infer_sampling_interval(series)
    span = series.index.max() - series.index.min() if len(series) else pd.Timedelta(0)
    keys = []
    if len(values) < 3 or dt is None or dt <= pd.Timedelta(0):
        return ["time_series"] if len(values) else keys
    keys += ["time_series", "availability", "histogram", "qq", "duration_curve"]
    sub_daily = dt < pd.Timedelta("1D")
    if sub_daily and span >= pd.Timedelta("1D"):
        keys.append("diurnal")
    if dt <= pd.Timedelta("1h") and span >= pd.Timedelta("2D"):
        keys.append("carpet")
    if values.index.to_period("M").nunique() >= 2:
        keys.append("monthly")
    if len(series) >= 20:
        keys.append("acf")
    if len(values) >= 64 and sub_daily:
        keys.append("psd")
    return keys


def figure_to_bytes(fig, fmt="png", dpi=300):
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=dpi)
    return buffer.getvalue()


def build_zip(files):
    """Pack ``{filename: bytes}`` into a ZIP archive and return its bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, content in files.items():
            archive.writestr(filename, content)
    return buffer.getvalue()
