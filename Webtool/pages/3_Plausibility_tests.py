import copy
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly_resampler import register_plotly_resampler, unregister_plotly_resampler

import data_plau as dplau
import data_processing as d_p
import data_filling as d_f
from Measurement import Measurement
from sympy.codegen.cfunctions import isnan

# Todo: add anomalie classes

register_plotly_resampler(mode="auto", default_n_shown_samples=50000)

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

# Main area content
st.title("Plausibility Tests")
st.write("Here you can check the time series for plausibility.")

st.markdown("""
### What are Plausibility Tests?
Plausibility tests help you verify the accuracy and consistency of your time series data. 
They ensure that the data meets certain criteria and is suitable for analysis.

### Steps to Perform a Plausibility Test:
1. **Upload your time series data** using the upload feature.
2. **Select the uploaded data** from the dropdown list.
3. **Choose the plausibility tests** you want to perform.
4. **Adjust the parameters** to fit your data.
5. **Click on the 'Start Plausibility Tests' button** to perform the plausibility tests.

It's an iterative process to choose the best parameters for the test. 
Try to start with small datasets and with the simpler tests.
""")

col5, col6 = st.columns(2)

# Add Measurement if coming from processing
if "measurement_add_button" in st.session_state and st.session_state["measurement_add_button"]:
    with st.spinner('Adding Timeseries...'):
        measurement_updated = st.session_state["measurement_updated"]
        # If no name is given, use filename + column name
        if st.session_state["measurement_add_name"] == "":
            st.session_state["measurement_add_name"] = measurement_updated.name + " processed"
        measurement_updated.name = st.session_state["measurement_add_name"]
        # Add to dict
        st.session_state["measurement_dict"][measurement_updated.name] = measurement_updated
    st.success(f'The time series {measurement_updated.name} has been successfully added.')

try:
    if "measurement_dict" in st.session_state:
        chosen_measurement_name = col5.selectbox("Choose time series", st.session_state["measurement_dict"])
        chosen_measurement = st.session_state["measurement_dict"][chosen_measurement_name]
        show_measurement = st.checkbox("Show time series?")
    else:
        st.session_state["measurement_dict"] = {}
        chosen_measurement = None
        show_measurement = None
except KeyError:
    st.error("No time series selected")

if chosen_measurement and show_measurement:
    pd.options.plotting.backend = "plotly"
    fig = chosen_measurement.return_df_as_datetime(raw=True)["value"].plot(title="", template="simple_white")

    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Messung",
                      plot_bgcolor="white", margin=dict(t=40, r=0, l=0))

    st.plotly_chart(fig, width='stretch')

tab1, tab2 = st.tabs(["Automatic Check", "Manual Check"])
col1, col2 = tab1.columns(2)

col1.write("Choose plausibility tests:")
check_gap = col1.checkbox('Gap', disabled=False,
                          help="https://gitlab.com/rhaberme/hsg_data_validation/-/wikis/gaps")
if check_gap:
    exp_ = col2.expander("Settings 'Gap'")
    with exp_:
        st.session_state['gap_custom_missing_values'] = st.text_input("Custom Missing Value")


check_constancy = col1.checkbox('Constancy', disabled=False,
                                    help=("https://gitlab.com/rhaberme/hsg_data_validation/-/wikis/check_constancy"))

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
                            help="https://gitlab.com/rhaberme/hsg_data_validation/-/wikis/check_range")
if check_range:
    exp_ = col2.expander("Settings 'Range'")
    with exp_:

        st.session_state['range_check_lower_border'] = st.number_input("Lower Border", value=0.0)
        st.session_state['range_check_upper_border'] = st.number_input("Upper Border", value=5.0)


check_outlier = col1.checkbox('Outlier', disabled=False, help="https://gitlab.com/rhaberme/hsg_data_validation/-/wikis/check_outlier")

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
                               help="https://gitlab.com/rhaberme/hsg_data_validation/-/wikis/check_gradient")

if check_gradient:
    exp_ = col2.expander("Settings 'Gradient'")
    with exp_:

        st.session_state['gradient_check_delta'] = st.number_input("Max. Delta per Timestep",
                                                                   min_value=0.00,
                                                                   max_value=10.00, value=1.50, step=0.01)

check_noise = col1.checkbox('Noise', disabled=False,
                            help="https://gitlab.com/rhaberme/hsg_data_validation/-/wikis/check_noise")
if check_noise:
    exp_ = col2.expander("Settings 'Noise'")
    with exp_:
        st.write("")

        st.session_state['noise_window_size'] = st.number_input("Window size of rolling std", min_value=1,
                                                                max_value=1000, value=10, step=1)
        st.session_state['noise_treshold'] = st.number_input("STD treshold", min_value=0.1,
                                                                max_value=1000.0, value=10.0, step=0.1)

check_drift = col1.checkbox('Drift', disabled=False,
                            help="https://gitlab.com/rhaberme/hsg_data_validation/-/wikis/check_drift")
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


check_jump = col1.checkbox('Jump', disabled=False,
                            help="https://gitlab.com/rhaberme/hsg_data_validation/-/wikis/check_jumps")
if check_jump:
    exp_ = col2.expander("Settings 'Jump'")
    with exp_:
        st.session_state['jump_window'] = st.number_input("Window for jump detection", min_value=1,
                                                                             max_value=1000, value=10, step=1 )
        threshold = st.session_state['jump_threshold'] =  st.number_input("Threshold for jump detection", min_value=0.0,
                                                                             max_value=100000.0, value=2.0, step=1.0)


col1.write("Plausibility Tests:")
fill_gaps = col1.checkbox('Filling data gaps?')
if fill_gaps:
    selected_fill_method = col1.selectbox("Select method for filling data gaps", ["Interpolation", "Regression",
                                                                                      "Null", "Average",
                                                                                      "Moving average"],
                                          key="selected_fill_method")
do_plausibility_checks = col1.button("Start plausibility tests")


if do_plausibility_checks:
    with (st.spinner("Data check is being performed")):
        df_raw = chosen_measurement.raw_df.copy()
        df_valid = chosen_measurement.raw_df.copy()
        df_valid["ABC"] = False # Range
        df_valid["DK"] = False # Gap
        df_valid["E"] = False # Constanz
        df_valid["F"] = False # Outlier
        df_valid["G"] = False # Gradient
        df_valid["H"] = False # Noise
        df_valid["I"] = False # Drift
        df_valid["J"] = False # Shift

        if check_gap:
            df_inplausible = dplau.check_gaps(df_raw,
                                              custom_missing_values=st.session_state["gap_custom_missing_values"])
            tab1.markdown(f"Plausibility test gap: {df_inplausible["Error"].sum()} inplausible data found.")
            df_valid.loc[df_inplausible["Error"], "DK"] = True
        #            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_range:
            df_inplausible = dplau.check_range(df_data=df_raw,
                                               upper_border=st.session_state['range_check_upper_border'],
                                               lower_border=st.session_state['range_check_lower_border'])
            tab1.markdown(f"plausibility test range: {df_inplausible["Error"].sum()} inplausible data found.")
            df_valid.loc[df_inplausible["Error"], "ABC"] = True
        #            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_constancy:
            df_inplausible = dplau.check_constancy(df_raw,
                                                   window=st.session_state["constancy_window"],
                                                   threshold=st.session_state["constancy_threshold"],
                                                   min_std=st.session_state["constancy_min_std"],
                                                   value_col_name="value",
                                                   method=st.session_state["constancy_method"])
            tab1.markdown(f"Plausibility test constancy: {df_inplausible["Error"].sum()} inplausible data found.")
            df_valid.loc[df_inplausible["Error"], "E"] = True
        #            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_outlier:
            df_inplausible = dplau.check_outlier(df_data=df_raw,
                                                 value_col_name="value",
                                                 method=st.session_state['outlier_method'],
                                                 std_multiplier=st.session_state['outlier_std_multiplier'],
                                                 iqr_multiplier=st.session_state['outlier_iqr_multiplier'])

            tab1.markdown(f"plausibility test outlier: {df_inplausible["Error"].sum()} inplausible data found.")
            df_valid.loc[df_inplausible["Error"], "F"] = True
        #            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_gradient:
            df_inplausible = dplau.check_gradients(df_data=df_raw,
                                                   value_col_name="value",
                                                   gradient_threshold=st.session_state['gradient_check_delta'])
            tab1.markdown(f"plausibility test gradient: {df_inplausible["Error"].sum()} inplausible data found.")
            df_valid.loc[df_inplausible["Error"], "G"] = True
        #            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_noise:
            df_inplausible = dplau.check_noise(df_data=df_raw,
                                               window_size=st.session_state['noise_window_size'],
                                               threshold=st.session_state['noise_treshold'],
                                               value_col="value")
            tab1.markdown(f"plausibility test noise: {df_inplausible["Error"].sum()} inplausible data found.")
            df_valid.loc[df_inplausible["Error"], "H"] = True
        #            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_drift:
            df_inplausible = dplau.check_drift(df_data=df_raw,
                                               window=st.session_state['drift_window_size'],
                                               threshold=st.session_state['drift_treshold'],
                                               zero=st.session_state['drift_zero'],
                                               method=st.session_state['drift_method'],
                                               value_col_name="value")
            tab1.markdown(f"plausibility test drift: {df_inplausible["Error"].sum()} inplausible data found.")
            df_valid.loc[df_inplausible["Error"], "I"] = True
        #            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        if check_jump:
            df_inplausible = dplau.check_jumps(df_data=df_raw,
                                               window=st.session_state['jump_window'],
                                               threshold=st.session_state['jump_threshold'],
                                               value_col_name="value")
            tab1.markdown(f"plausibility test jumps: {df_inplausible["Error"].sum()} inplausible data found.")
            df_valid.loc[df_inplausible["Error"], "J"] = True
        #            df_raw = d_p.drop_implausible_measurements(df_raw, df_inplausible, inplausible_column_name="Error")

        df_raw = d_p.drop_implausible_measurements(df_raw, df_valid, inplausible_column_name="ABC")
        df_raw = d_p.drop_implausible_measurements(df_raw, df_valid, inplausible_column_name="DK")
        df_raw = d_p.drop_implausible_measurements(df_raw, df_valid, inplausible_column_name="E")
        df_raw = d_p.drop_implausible_measurements(df_raw, df_valid, inplausible_column_name="F")
        df_raw = d_p.drop_implausible_measurements(df_raw, df_valid, inplausible_column_name="G")
        df_raw = d_p.drop_implausible_measurements(df_raw, df_valid, inplausible_column_name="H")
        df_raw = d_p.drop_implausible_measurements(df_raw, df_valid, inplausible_column_name="I")
        df_raw = d_p.drop_implausible_measurements(df_raw, df_valid, inplausible_column_name="J")

        if fill_gaps and "selected_fill_method" in st.session_state.keys():
            df_filled = d_f.data_filling_fun_dict[selected_fill_method](df_raw) if selected_fill_method else df_raw
        else:
            df_filled = df_raw.copy()

        df_filled = d_p.df_to_datetime(df_filled)

        df_validated = df_filled.copy()
        st.session_state["changed_df"] = df_validated

        indices_of_nan = pd.isnull(df_raw).any(axis=1).to_numpy().nonzero()[0]

        indices_of_ABC = indices_of_nan[df_valid["ABC"].iloc[indices_of_nan]]
        indices_of_DK = indices_of_nan[df_valid["DK"].iloc[indices_of_nan]]
        indices_of_E = indices_of_nan[df_valid["E"].iloc[indices_of_nan]]
        indices_of_F = indices_of_nan[df_valid["F"].iloc[indices_of_nan]]
        indices_of_G = indices_of_nan[df_valid["G"].iloc[indices_of_nan]]
        indices_of_H = indices_of_nan[df_valid["H"].iloc[indices_of_nan]]
        indices_of_I = indices_of_nan[df_valid["I"].iloc[indices_of_nan]]
        indices_of_J = indices_of_nan[df_valid["J"].iloc[indices_of_nan]]


        df_ABC = df_raw.copy()
        df_ABC["value"] = np.nan
        df_ABC = d_p.df_to_datetime(df_ABC)
        df_ABC = d_p.drop_duplicated_indices(df_ABC)
        df_DK = df_ABC.copy()
        df_E = df_ABC.copy()
        df_F = df_ABC.copy()
        df_G = df_ABC.copy()
        df_H = df_ABC.copy()
        df_I = df_ABC.copy()
        df_J = df_ABC.copy()
        df_ABC.iloc[indices_of_ABC] = chosen_measurement.raw_df.iloc[indices_of_ABC]
        df_DK.iloc[indices_of_DK] = chosen_measurement.raw_df.iloc[indices_of_DK]
        df_E.iloc[indices_of_E] = chosen_measurement.raw_df.iloc[indices_of_E]
        df_F.iloc[indices_of_F] = chosen_measurement.raw_df.iloc[indices_of_F]
        df_G.iloc[indices_of_G] = chosen_measurement.raw_df.iloc[indices_of_G]
        df_H.iloc[indices_of_H] = chosen_measurement.raw_df.iloc[indices_of_H]
        df_I.iloc[indices_of_I] = chosen_measurement.raw_df.iloc[indices_of_I]
        df_J.iloc[indices_of_J] = chosen_measurement.raw_df.iloc[indices_of_J]

        tab1.write("Time series after validation:")
        changed_df_with_deleted_and_new = df_filled.copy()
        if "status" in changed_df_with_deleted_and_new:
            changed_df_with_deleted_and_new.drop(["status"], axis=1, inplace=True)

        changed_df_with_deleted_and_new = d_p.df_to_datetime(changed_df_with_deleted_and_new)
        if check_range:
            changed_df_with_deleted_and_new["ABC"] = df_ABC.value
        if check_gap:
            changed_df_with_deleted_and_new["DK"] = df_DK.value
        if check_constancy:
            changed_df_with_deleted_and_new["E"] = df_E.value
        if check_outlier:
            changed_df_with_deleted_and_new["F"] = df_F.value
        if check_gradient:
            changed_df_with_deleted_and_new["G"] = df_G.value
        if check_noise:
            changed_df_with_deleted_and_new["H"] = df_H.value
        if check_drift:
            changed_df_with_deleted_and_new["I"] = df_I.value
        if check_jump:
            changed_df_with_deleted_and_new["J"] = df_J.value
        chosen_measurement.changed_df_with_deleted_and_new = changed_df_with_deleted_and_new
        if hasattr(chosen_measurement, "days_changed_dict"):
            del chosen_measurement.__dict__["days_changed_dict"]
        st.session_state["validation_check_iteration_nr"] = 0
        chosen_measurement.validated_df = st.session_state["changed_df"]

        tab1.success(f'The validated data was added to the time series '
                     f'{chosen_measurement.name}.')

        # http://hclwizard.org:3000/hclwizard/
        # can be updated if there are better ideas
        anomaly_print_dict = {
            "ABC" : {"name": "Range (ABC)", "color": "#DB9D85"},
#            "B": "#E093C3",
#            "C": "#ACA4E2",
            "DK" : {"name": "Gap (DK)", "color": "#4CB9CC"},
#            "K": "#4CB9CC"
            "E" : {"name": "Constancy (E)", "color": "#5CBD92"},
            "F" : {"name": "Outliers (F)", "color": "#ABB065"},
            "G" : {"name": "Gradient (G)", "color": "#DB9D85"},
            "H" : {"name": "Noise (H)", "color": "#89D9CF"},
            "I" : {"name": "Drift (I)", "color": "#AC7F21"},
            "J" : {"name": "Jump (J)", "color": "#533600"},
        }

        pd.options.plotting.backend = "plotly"
        fig = go.Figure()
        fig = fig.add_trace(go.Scatter(
            x = changed_df_with_deleted_and_new.index,
            y=changed_df_with_deleted_and_new['value'],
            name=f"Validated time series {chosen_measurement.name}"
        ))
        for name, values in changed_df_with_deleted_and_new.items():
            if name == "value":
                continue
            fig = fig.add_trace(go.Scatter(
                x=values.index,
                y=values,
                name=anomaly_print_dict[name]['name'],
                mode="markers",
                marker=dict(color=anomaly_print_dict[name]['color'],),
            ))
        fig.update_layout(showlegend=True, xaxis_title="", yaxis_title="Messung",
                          plot_bgcolor="white", margin=dict(t=40, r=1, l=1))

        tab1.plotly_chart(fig, width='stretch')

        measurement_add_col1, measurement_add_col2 = tab1.columns([2, 1])
        measurement_add_col1.text_input("Time series name:", placeholder=f"Enter time series name. Defaults to adding '{chosen_measurement.name} processed'.", key="measurement_add_name")
        measurement_add_col2.button(label='Add to selectable timeseries', key="measurement_add_button")

        #save updated measurement
        measurement_updated = copy.deepcopy(chosen_measurement)
        measurement_updated.raw_df = measurement_updated.validated_df.copy()
        st.session_state["measurement_updated"] = measurement_updated

        @st.cache_data
        def convert_for_download(df):
            return df.to_csv(na_rep="nan").encode("utf-8")

        # Assemble df for download
        df_download = chosen_measurement.validated_df.copy()
        for name, values in changed_df_with_deleted_and_new.items():
            if name == "value":
                continue
            df_download[anomaly_print_dict[name]['name']] = values.map(lambda x: not pd.isna(x))

        csv = convert_for_download(df_download)

        st.download_button(
            label="Download validated time series",
            data=csv,
            file_name="validated_timeseries.csv",
            mime="text/csv",
            icon=":material/download:",
        )

tab2.markdown('<p class="small-font-red">Manual data verification is still under development!</p>',
              unsafe_allow_html=True)


