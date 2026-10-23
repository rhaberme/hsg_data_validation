import json

import pandas as pd
import streamlit as st

import scientific_plots as sp

st.set_page_config(
    page_title="HSG Sim Datatool",
    page_icon="HSG",
    layout="centered"
)

hide_menu = """
<style>
#MainMenu {visibility:hidden;}
footer{visibility:hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

st.title("Scientific Plots")


PLOT_INFO = {
    "time_series": {
        "title": "Time series",
        "caption": "Time series of {var} from {start} to {end} (sampling interval {dt}). "
                   "Ticks at the bottom mark missing values.",
    },
    "availability": {
        "title": "Data availability",
        "caption": "Data completeness of {var} per {period}. Periods below the dashed 90 % line are highlighted.",
    },
    "histogram": {
        "title": "Distribution",
        "caption": "Probability density of {var} (n = {n}): histogram (Freedman–Diaconis bins), "
                   "Gaussian kernel density estimate (KDE, Scott's bandwidth) and fitted normal distribution.",
    },
    "qq": {
        "title": "Normal Q-Q plot",
        "caption": "Normal quantile-quantile plot of {var}. Deviations from the reference line "
                   "indicate departures from normality.",
    },
    "duration_curve": {
        "title": "Duration curve",
        "caption": "Duration curve of {var}: values sorted in descending order against their exceedance "
                   "probability (Weibull plotting position).",
    },
    "diurnal": {
        "title": "Diurnal profile",
        "caption": "Diurnal profile of {var}: median, mean, interquartile range and 5–95 % range per hour of day "
                   "(times in UTC).",
    },
    "carpet": {
        "title": "Carpet plot",
        "caption": "Carpet plot of hourly mean values of {var} (times in UTC). White cells indicate missing data.",
    },
    "monthly": {
        "title": "Monthly box plot",
        "caption": "Monthly distribution of {var}. Boxes show the interquartile range, the line the median "
                   "and whiskers the 5th and 95th percentiles.",
    },
    "acf": {
        "title": "Autocorrelation",
        "caption": "Sample autocorrelation function of {var} (pairwise-complete observations). The shaded band "
                   "is the approximate 95 % confidence interval of white noise (±1.96/√n).",
    },
    "psd": {
        "title": "Power spectrum",
        "caption": "Power spectral density of {var} estimated with Welch's method (linear detrending; "
                   "gaps linearly interpolated). Dotted lines mark common periodicities.",
    },
}

if not st.session_state.get("measurement_dict"):
    st.info("No time series available. Please upload a time series on the page **Add time series** first.")
    st.stop()


col1, col2 = st.columns(2)
chosen_name = col1.selectbox("Choose time series", list(st.session_state["measurement_dict"]))
measurement = st.session_state["measurement_dict"][chosen_name]

has_validated = measurement.validated_df is not None
data_options = ["Raw data"] + (["Validated data"] if has_validated else [])
data_source = col2.radio("Data", data_options, index=len(data_options) - 1, horizontal=True,
                         help="Validated data is available after running the plausibility tests.")

raw_series = measurement.return_df_as_datetime(raw=True)["value"].astype(float)
if data_source == "Validated data":
    series = measurement.validated_df["value"].astype(float)
    series.index = pd.DatetimeIndex(series.index)
else:
    series = raw_series

with st.expander("Figure settings", expanded=False):
    s1, s2 = st.columns(2)
    variable_label = s1.text_input("Axis label (with unit)", value=str(measurement.label_value),
                                   help="E.g. 'Water level [cm]'. Used for all axes and captions.")
    width_label = s2.selectbox("Figure width", list(sp.FIGURE_WIDTHS_CM), index=2)
    font_size = s1.number_input("Font size [pt]", min_value=5, max_value=16, value=8, step=1)
    font_family = s2.selectbox("Font family", ["sans-serif", "serif"])
    file_format = s1.selectbox("Export format", ["pdf", "svg", "png"],
                               help="PDF and SVG are vector formats and recommended for publications.")
    dpi = s2.number_input("Resolution for PNG [dpi]", min_value=72, max_value=1200, value=300, step=50)
    compare_raw = st.checkbox("Show raw data in background of the time series plot",
                              value=data_source == "Validated data", disabled=data_source != "Validated data")
    log_duration = st.checkbox("Logarithmic y axis for the duration curve")

width_cm = sp.FIGURE_WIDTHS_CM[width_label]
# half-width figures (distribution etc.) are placed side by side in double-column layouts
small_width_cm = width_cm / 2 if width_cm > 12 else width_cm

available = sp.available_plots(series)
if not available:
    st.warning("The selected time series contains no valid values.")
    st.stop()

selected = st.multiselect("Figures", available, default=available,
                          format_func=lambda key: PLOT_INFO[key]["title"])


def _make_figure(key, series, raw_series, label, width_cm, small_width_cm, compare_raw, log_duration):
    if key == "time_series":
        return sp.plot_time_series(series, label, width_cm,
                                   comparison=raw_series if compare_raw else None,
                                   series_label="Validated" if compare_raw else label)
    if key == "availability":
        return sp.plot_data_availability(series, width_cm)
    if key == "histogram":
        return sp.plot_histogram(series, label, small_width_cm)
    if key == "qq":
        return sp.plot_qq(series, small_width_cm)
    if key == "duration_curve":
        return sp.plot_duration_curve(series, label, small_width_cm, log_y=log_duration)
    if key == "diurnal":
        return sp.plot_diurnal_profile(series, label, small_width_cm)
    if key == "carpet":
        return sp.plot_carpet(series, label, width_cm)
    if key == "monthly":
        return sp.plot_monthly_boxplot(series, label, width_cm)
    if key == "acf":
        return sp.plot_acf(series, small_width_cm)
    if key == "psd":
        return sp.plot_power_spectrum(series, label, small_width_cm)
    raise KeyError(key)


@st.cache_data(show_spinner=False, max_entries=64)
def render_plot(key, series, raw_series, label, width_cm, small_width_cm, font_size, font_family,
                file_format, dpi, compare_raw, log_duration):
    with sp.plot_style(font_size, font_family):
        fig = _make_figure(key, series, raw_series, label, width_cm, small_width_cm, compare_raw, log_duration)
        preview = sp.figure_to_bytes(fig, "png", dpi=200)
        export = preview if file_format == "png" and dpi == 200 else sp.figure_to_bytes(fig, file_format, dpi)
    return preview, export


@st.cache_data(show_spinner=False)
def compute_statistics(series):
    return sp.descriptive_statistics(series)


dt = sp.infer_sampling_interval(series)
span = series.index.max() - series.index.min()
caption_values = {
    "var": variable_label,
    "start": series.index.min().strftime("%Y-%m-%d %H:%M"),
    "end": series.index.max().strftime("%Y-%m-%d %H:%M"),
    "dt": sp._format_timedelta(dt),
    "n": f"{int(series.notna().sum()):,}",
    "period": "day" if span <= pd.Timedelta("400D") else "month",
}

safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in chosen_name).strip("_") or "timeseries"
export_files = {}
captions = {}


st.subheader("Descriptive statistics")
statistics = compute_statistics(series)
st.table(statistics.map(lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)))
export_files[f"{safe_name}_statistics.csv"] = statistics.to_csv().encode("utf-8")


st.subheader("Figures")
progress = st.progress(0.0, text="Creating figures...") if selected else None
for i, key in enumerate(selected):
    info = PLOT_INFO[key]
    try:
        preview, export = render_plot(key, series, raw_series if compare_raw else None, variable_label,
                                      width_cm, small_width_cm, font_size, font_family, file_format, dpi,
                                      compare_raw and data_source == "Validated data", log_duration)
    except Exception as error:  # a single failing figure must not break the page
        st.warning(f"Figure '{info['title']}' could not be created: {error}")
        continue
    finally:
        progress.progress((i + 1) / len(selected), text=f"Creating figures... ({i + 1}/{len(selected)})")

    caption = info["caption"].format(**caption_values)
    filename = f"{safe_name}_{key}.{file_format}"
    export_files[filename] = export
    captions[filename] = caption

    st.markdown(f"**Figure {i + 1}: {info['title']}**")
    st.image(preview, width="stretch" if key in ("time_series", "availability", "carpet", "monthly") else "content")
    c1, c2 = st.columns([4, 1])
    c1.caption(caption)
    c2.download_button(f"{file_format.upper()}", data=export, file_name=filename, key=f"download_{key}",
                       icon=":material/download:")
if progress is not None:
    progress.empty()


metadata = {
    "time_series": chosen_name,
    "data": data_source,
    "variable": variable_label,
    "start": caption_values["start"],
    "end": caption_values["end"],
    "sampling_interval": caption_values["dt"],
    "figure_width_cm": width_cm,
    "font_size_pt": font_size,
    "format": file_format,
    "dpi": dpi if file_format == "png" else None,
    "created_with": "HSG Sim Datatool (https://gitlab.com/rhaberme/hsg_data_validation)",
}
export_files["captions.txt"] = "\n\n".join(f"{name}\n{text}" for name, text in captions.items()).encode("utf-8")
export_files["metadata.json"] = json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")

st.download_button("Download all figures, statistics and captions (ZIP)", data=sp.build_zip(export_files),
                   file_name=f"{safe_name}_scientific_plots.zip", mime="application/zip",
                   icon=":material/folder_zip:", type="primary")
