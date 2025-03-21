import streamlit as st
import pandas as pd
import numpy as np
from plotly_resampler import register_plotly_resampler, unregister_plotly_resampler

import data_plau as dplau
import data_processing as d_p
import data_filling as d_f


data_filling_fun_dict = {"Null": d_f.fill_nan_null,
                         "Interpolation": d_f.fill_nan_interp,
                         "Regression": None,
                         "Mittelwert": d_f.fill_nan_mean,
                         "Gleitendes Mittel": d_f.fill_nan_rollmean}

register_plotly_resampler(mode="auto", default_n_shown_samples=50000)

st.set_page_config(
    page_title="CSO Data-Tool",
    page_icon="🌊",
    layout="centered"
)

hide_menu = """
<style>
#MainMenu {visibility:hidden;}
footer{visibility:hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)
st.markdown("""
<style>
.small-font-green {
    font-size:12px;
    color: green;
}
.small-font-red {
    font-size:12px;
    color: red;
}
.normal-font-red {
    font-size:16px;
    color: red;
}
.normal-font-green {
    font-size:16px;
    color: green;
}
</style>
""", unsafe_allow_html=True)




st.write("""
# Plausibility Tests
"""
         )
st.sidebar.markdown("Checking the timeseries for plausibility")

col5, col6 = st.columns(2)

try:
    if "measurement_dict" in st.session_state:
        chosen_measurement_name = col5.selectbox("Choose Timeseries", st.session_state["measurement_dict"])
        chosen_measurement = st.session_state["measurement_dict"][chosen_measurement_name]
        show_measurement = col6.select_slider("Show Timeseries:", ["", " "])
    else:
        st.session_state["measurement_dict"] = {}
        chosen_measurement = None
        show_measurement = None
except KeyError:
    st.error("Keine Messreihe ausgewählt")

if chosen_measurement and show_measurement == " ":
    pd.options.plotting.backend = "plotly"
    fig = chosen_measurement.return_df_as_datetime(raw=True)["value"].plot(title="", template="simple_white")

    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Messung",
                      plot_bgcolor="white", margin=dict(t=40, r=0, l=0))

    st.plotly_chart(fig, use_container_width=True)

tab1, tab2 = st.tabs(["Automatic Check", "Manual Check"])
col1, col2 = tab1.columns(2)

col1.write("Choose plausibility tests:")
check_gap = col1.checkbox('Gap', disabled=False,
                          help="")
if check_gap:
    exp_ = col2.expander("Settings 'Gap'")
    with exp_:
        st.session_state['gap_custom_missing_values'] = st.text_input("Custom Missing Value")


check_constancy = col1.checkbox('Constancy', disabled=False,
                                    help=(""))

if check_constancy:
    exp_ = col2.expander("Settings 'Constancy'")
    with exp_:
        st.markdown("""

        """)
        # window=2, threshold=1e-8, min_std=0, method="threshold"
        constancy_methods = ["threshold", "std"]
        st.session_state['constancy_method'] = st.selectbox("Constancy Test Methode", constancy_methods)
        if st.session_state['constancy_method'] == "threshold":
            st.session_state['constancy_window'] = st.number_input("Window of the rolling difference", min_value=1,
                                                         max_value=10000, value=2, step=1)
            #st.session_state['threshold'] = st.number_input("Constancy Treshold", min_value=1e-10, max_value=1e-6, value=1e-8)
            st.session_state['constancy_threshold'] = 1e-8
            st.session_state['constancy_min_std'] = 0
        else:
            st.session_state['constancy_window'] = st.number_input("Window of the rolling difference", min_value=1,
                                                         max_value=10000, value=2, step=1)
            st.session_state['constancy_threshold'] = 0
            st.session_state['constancy_min_std'] = st.number_input("Min. STD", min_value=0.01,
                                                         max_value=100000.0, value=2.0)

check_range = col1.checkbox('Range',
                            help="")
if check_range:
    exp_ = col2.expander("Settings 'Range'")
    with exp_:

        st.session_state['range_check_lower_border'] = st.number_input("Lower Border", value=0.0)
        st.session_state['range_check_upper_border'] = st.number_input("Upper Border", value=5.0)


check_outlier = col1.checkbox('Outlier', disabled=False, help="")

if check_outlier:
    exp_ = col2.expander("Settings 'Outlier'")
    with exp_:
        st.markdown("""
       
        """)
        # method="std_method" std_multiplier=3, iqr_multiplier=1.5
        outlier_methods = ["iqr_method", "std_method"]
        st.session_state['outlier_method'] = st.selectbox("Outlier Test Methode", outlier_methods)
        if st.session_state['outlier_method'] == "std_method":
            st.session_state['outlier_std_multiplier'] = st.number_input("STD Multiplier", min_value=0.1,
                                                         max_value=1000.0, value=3.0, step=0.1)
            st.session_state['outlier_iqr_multiplier'] = None
        else:
            st.session_state['outlier_iqr_multiplier'] = st.number_input("IQR Multiplier", min_value=0.1,
                                                         max_value=1000.0, value=1.5, step=0.1)
            st.session_state['outlier_std_multiplier'] = None

check_gradient = col1.checkbox('Gradient', disabled=False,
                               help="")

if check_gradient:
    exp_ = col2.expander("Settings 'Gradient'")
    with exp_:

        st.session_state['gradient_check_delta'] = st.number_input("Max. Delta per Minute",
                                                                   min_value=0.00,
                                                                   max_value=10.00, value=1.50, step=0.01)

check_noise = col1.checkbox('Noise', disabled=False,
                            help="")
if check_noise:
    exp_ = col2.expander("Settings 'Noise'")
    with exp_:
        st.write("")

        st.session_state['noise_window_size'] = st.number_input("Window size of rolling std", min_value=1,
                                                                max_value=1000, value=10, step=1)
        st.session_state['noise_treshold'] = st.number_input("STD treshold", min_value=0.1,
                                                                max_value=1000.0, value=10.0, step=0.1)

check_drift = col1.checkbox('Drift', disabled=False,
                            help="")
if check_drift:
    exp_ = col2.expander("Settings 'Drift'")
    with exp_:
        drift_methods = ["mean", "zero"]
        st.session_state['drift_method'] = st.selectbox("Drift Detection Method", drift_methods)

        if st.session_state['drift_method'] == "mean":
            window=st.session_state['drift_window_size'] = st.number_input("Window for rolling mean", min_value=1,
                                                                           max_value=1000, value=10, step=1)
            threshold = st.session_state['drift_treshold'] = st.number_input("Treshold for rolling mean", min_value=0.1,
                                                                             max_value=100.0, value=1.0, step=0.1)
            st.session_state['drift_zero'] = None
        else:
            window = st.session_state['drift_window_size'] = st.number_input("Window for rolling mean", min_value=1,
                                                                             max_value=1000, value=10, step=1)
            threshold=st.session_state['drift_treshold'] = st.number_input("Treshold for rolling mean", min_value=0.1,
                                                                             max_value=100.0, value=1.0, step=0.1)
            st.session_state['drift_zero'] = st.number_input("Zero", min_value=-1000.0, max_value=1000.0, value=0.0,
                                                             step=0.1)

col1.write("Plausibility Tests:")
fill_gaps = col1.checkbox('Datenlücken füllen?')
if fill_gaps:
    selected_fill_method = col1.selectbox("Methode zur Füllung der Daten auswählen", ["Interpolation", "Regression",
                                                                                      "Null", "Mittelwert",
                                                                                      "Gleitendes Mittel"],
                                          key="selected_fill_method")
do_plausibility_checks = col1.button("Started plausibility tests")
if do_plausibility_checks:
    with (st.spinner("Datenprüfung wird ausgeführt")):
        df_raw = chosen_measurement.raw_df.copy()
        if check_gap:
            df_inplausible = dplau.check_gaps(df_raw,
                                              custom_missing_values=st.session_state["gap_custom_missing_values"])
            tab1.markdown(f"Plausibility test gap: {df_inplausible["Error"].sum()} inplausible data found.")
            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_constancy:
            df_inplausible = dplau.check_constancy(df_raw,
                                                   window=st.session_state["constancy_window"],
                                                   threshold=st.session_state["constancy_threshold"],
                                                   min_std=st.session_state["constancy_min_std"],
                                                   value_col_name="value",
                                                   method=st.session_state["constancy_method"])
            tab1.markdown(f"Plausibility test constancy: {df_inplausible["Error"].sum()} inplausible data found.")
            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_range:
            df_inplausible = dplau.check_range(df_data=df_raw,
                                               upper_border=st.session_state['range_check_upper_border'],
                                               lower_border=st.session_state['range_check_lower_border'])
            tab1.markdown(f"plausibility test range: {df_inplausible["Error"].sum()} inplausible data found.")

            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_outlier:
            df_inplausible = dplau.check_outlier(df_data=df_raw,
                                                 value_col_name="value",
                                                 method=st.session_state['outlier_method'],
                                                 std_multiplier=st.session_state['outlier_std_multiplier'],
                                                 iqr_multiplier=st.session_state['outlier_iqr_multiplier'])

            tab1.markdown(f"plausibility test outlier: {df_inplausible["Error"].sum()} inplausible data found.")
            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_gradient:
            df_inplausible = dplau.check_gradients(df_data=df_raw,
                                                   value_col_name="value",
                                                   gradient_threshold=st.session_state['gradient_check_delta'])
            tab1.markdown(f"plausibility test gradient: {df_inplausible["Error"].sum()} inplausible data found.")
            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_noise:
            df_inplausible = dplau.check_noise(df_data=df_raw,
                                               window_size=st.session_state['noise_window_size'],
                                               threshold=st.session_state['noise_treshold'],
                                               value_col="value")
            tab1.markdown(f"plausibility test noise: {df_inplausible["Error"].sum()} inplausible data found.")
            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_drift:
            df_inplausible = dplau.check_drift(df_data=df_raw,
                                               window=st.session_state['drift_window_size'],
                                               threshold=st.session_state['drift_treshold'],
                                               zero=st.session_state['drift_zero'],
                                               method=st.session_state['drift_method'],
                                               value_col_name="value")
            tab1.markdown(f"plausibility test drift: {df_inplausible["Error"].sum()} inplausible data found.")
            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")


        if "selected_fill_method" in st.session_state.keys() and st.session_state[
            "selected_fill_method"] != "Regression":
            df_filled = data_filling_fun_dict[selected_fill_method](
                df_raw) if selected_fill_method else df_raw
        else:
            df_filled = df_raw.copy()

        df_validated = df_raw.copy()
        st.session_state["changed_df"] = df_validated

        df_nan = df_raw.copy()
        df_nan["value"] = np.nan
        indices_of_nan = pd.isnull(df_raw).any(axis=1).to_numpy().nonzero()[0]

        df_nan.iloc[indices_of_nan] = chosen_measurement.raw_df.iloc[indices_of_nan]
        df_nan = d_p.df_to_datetime(df_nan)
        df_nan = d_p.drop_duplicated_indices(df_nan)

        tab1.write("Timeseries after validation:")
        changed_df_with_deleted_and_new = df_raw.copy()
        if "status" in changed_df_with_deleted_and_new:
            changed_df_with_deleted_and_new.drop(["status"], axis=1, inplace=True)

        changed_df_with_deleted_and_new = d_p.df_to_datetime(changed_df_with_deleted_and_new)
        changed_df_with_deleted_and_new["Deleted Data"] = df_nan.copy().value
        chosen_measurement.changed_df_with_deleted_and_new = changed_df_with_deleted_and_new
        if hasattr(chosen_measurement, "days_changed_dict"):
            del chosen_measurement.__dict__["days_changed_dict"]
        st.session_state["validation_check_iteration_nr"] = 0
        chosen_measurement.validated_df = st.session_state["changed_df"]

        tab1.success(f'Die validierten Daten wurden zur Messreihe '
                     f'{chosen_measurement.name} hinzugefügt.')

        pd.options.plotting.backend = "plotly"
        fig = changed_df_with_deleted_and_new.plot(title="", template="simple_white")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Messung",
                          plot_bgcolor="white", margin=dict(t=40, r=1, l=1))

        tab1.plotly_chart(fig, use_container_width=True)


        fig2 = df_filled.plot(title="", template="simple_white")
        fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="Messung",
                          plot_bgcolor="white", margin=dict(t=40, r=1, l=1))

        tab1.plotly_chart(fig2, use_container_width=True)

tab2.markdown('<p class="small-font-red">Händische Datenprüfung ist noch in der Entwicklung!</p>',
              unsafe_allow_html=True)
