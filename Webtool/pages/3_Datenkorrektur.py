import streamlit as st
import pandas as pd
import numpy as np
import data_plau as dplau
import data_processing as d_p
import data_filling as d_f
from plotly_resampler import register_plotly_resampler, unregister_plotly_resampler

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



data_filling_fun_dict = {"Null": d_f.fill_nan_null,
                         "Interpolation": d_f.fill_nan_null,
                         "Regression": None,
                         "Mittelwert": d_f.fill_nan_mean,
                         "Gleitendes Mittel": d_f.fill_nan_rollingmean}

st.write("""
# Prüfung und Korrektur
"""
         )
st.sidebar.markdown("Prüfung und Korrektur der Messreihe️")

col5, col6 = st.columns(2)

if "measurement_dict" in st.session_state:
    chosen_measurement_name = col5.selectbox("Messreihe auswählen", st.session_state["measurement_dict"])
    chosen_measurement = st.session_state["measurement_dict"][chosen_measurement_name]
    show_measurement = col6.select_slider("Graph anzeigen:", ["", " "])
else:
    st.session_state["measurement_dict"] = {}
    chosen_measurement = None
    show_measurement = None


if chosen_measurement and show_measurement == " ":
    pd.options.plotting.backend = "plotly"
    fig = chosen_measurement.return_df_as_datetime(raw=True)["value"].plot(title="", template="simple_white")
    suffix_dict = {"Höhenstand": " m", "Beckenüberlauf": " l/s", "Klärüberlauf": " l/s",
                   "Niederschlag": " mm", "Durchfluss": " l/s"}

    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Messung",
                      plot_bgcolor="white", margin=dict(t=40, r=0, l=0))

    st.plotly_chart(fig, use_container_width=True)

tab1, tab2 = st.tabs(["Automatische Prüfung", "Händische Prüfung"])
col1, col2 = tab1.columns(2)

if chosen_measurement and not chosen_measurement.status_available:
    kick_status = col1.checkbox('Prüfung anhand Statuscode', disabled=True,
                                help="Kein Statuscode verfügbar.")
else:
    kick_status = col1.checkbox('Prüfung anhand Statuscode')

col1.write("Plausibilitätstests auswählen:")
check_gap = col1.checkbox('Lücke', disabled=True,
                          help="")
if check_gap:
    if chosen_measurement and chosen_measurement.measurement_type == 'Höhenstand':
        unit = "[m]"
    elif chosen_measurement and chosen_measurement.measurement_type == "Niederschlag":
        unit = "[mm]"
    elif not chosen_measurement:
        unit = ""
    else:
        unit = "[l/s]"
    exp_ = col2.expander("Einstellungen 'Lücke'")
    with exp_:
        st.session_state['gap_check_min_border'] = st.number_input("Mindestwert Messung " + unit, min_value=-5.00,
                                                                   max_value=5.00, value=0.00, step=0.01)

if chosen_measurement and chosen_measurement.measurement_type == '':
    check_constancy = col1.checkbox('Konstanz', disabled=True,
                                    help=(""))
else:
    check_constancy = col1.checkbox('Konstanz', disabled=True,
                                    help=(""))

if check_constancy:
    exp_ = col2.expander("Einstellungen 'Konstanz'")
    with exp_:
        if chosen_measurement and chosen_measurement.measurement_type == 'Höhenstand':
            unit = "[m]"
        elif chosen_measurement and chosen_measurement.measurement_type == "Niederschlag":
            unit = "[mm]"
        elif not chosen_measurement:
            unit = ""
        else:
            unit = "[l/s]"
        st.session_state['check_constancy_nr_sequentiel_measurments'] = \
            st.number_input("Anzahl n der sequenziellen Zeitschritte",
                            min_value=2,
                            max_value=15, value=3, step=1)
        st.session_state['check_constancy_sum_delta'] = \
            st.number_input("Betrag der maximalen Wertänderung in n Zeitschritten " + unit,
                            min_value=3.00,
                            max_value=100.00, value=10.0, step=0.01)

check_range = col1.checkbox('Spanne',
                            help="")
if check_range:
    if chosen_measurement and chosen_measurement.measurement_type == 'Höhenstand':
        unit = "[m]"
    elif chosen_measurement and chosen_measurement.measurement_type == "Niederschlag":
        unit = "[mm]"
    elif not chosen_measurement:
        unit = ""
    else:
        unit = "[l/s]"
    exp_ = col2.expander("Einstellungen 'Spanne'")
    with exp_:

        st.session_state['range_check_lower_border'] = st.number_input("Unterer Grenzwert " + unit, max_value=5.00,
                                                                       value=0.00,
                                                                       step=0.01)
        st.session_state['range_check_upper_border'] = st.number_input("Oberer Grenzwert " + unit, min_value=-100.00,
                                                                       max_value=20.00, value=4.00, step=0.01)

        # st.session_state['r1_check_outlier'], st.session_state['r3_check_outlier'] = \
        #    st.slider('Erlaubte Messwertspanne', min_value=-5.0, max_value=20.0, value=(0.2, 4.0), step=0.1)
if chosen_measurement and chosen_measurement.measurement_type == 'Höhenstand':
    check_outlier = col1.checkbox('Ausreißer', disabled=True,
                                  help="")
else:
    check_outlier = col1.checkbox('Ausreißer', disabled=True,
                                  help="")

if check_outlier:
    exp_ = col2.expander("Einstellungen 'Ausreißer'")
    with exp_:
        st.markdown("""
       
        """)

        st.session_state['r1_check_outlier'] = round(st.number_input("Unterer Perzentil", min_value=1, max_value=50,
                                                                     value=5,
                                                                     step=1) / 100, 2)
        st.session_state['r3_check_outlier'] = round(st.number_input("Oberer Perzentil", min_value=51,
                                                                     max_value=100, value=95, step=1) / 100, 2)
if chosen_measurement and chosen_measurement.measurement_type == 'Niederschlag':
    check_gradient = col1.checkbox('Gradient', disabled=True,
                                   help="")
else:
    check_gradient = col1.checkbox('Gradient', disabled=False,
                                   help="")
if check_gradient:
    exp_ = col2.expander("Einstellungen 'Gradient'")
    with exp_:
        if chosen_measurement and chosen_measurement.measurement_type == 'Höhenstand':
            unit = "[m]"
        elif chosen_measurement and chosen_measurement.measurement_type == "Niederschlag":
            unit = "[mm]"
        elif not chosen_measurement:
            unit = ""
        else:
            unit = "[l/s]"

        st.session_state['gradient_check_delta'] = st.number_input("Maximale Wertänderung je Minute " + unit,
                                                                   min_value=-100.00,
                                                                   max_value=10.00, value=1.50, step=0.01)

check_noise = col1.checkbox('Rauschen', disabled=True,
                            help="")
if check_noise:
    exp_ = col2.expander("Einstellungen 'Rauschen'")
    with exp_:
        st.write("")

check_drift = col1.checkbox('Drift', disabled=True,
                            help="")
if check_drift:
    exp_ = col2.expander("Einstellungen 'Drift'")
    with exp_:
        st.session_state["check_drift_chunk_split"] = st.number_input("Aufteilung 'Chunk-Split'", min_value=2, value=6)

col1.write("Datenergänzung und Korrektur:")
fill_gaps = col1.checkbox('Datenlücken füllen?')
if fill_gaps:
    selected_fill_method = col1.selectbox("Methode zur Füllung der Daten auswählen", ["Interpolation", "Regression",
                                                                                      "Null", "Mittelwert",
                                                                                      "Gleitendes Mittel"],
                                          key="selected_fill_method")

do_plausibility_checks = col1.button("Korrektur starten")
if do_plausibility_checks:
    with st.spinner("Datenprüfung wird ausgeführt"):
        try:
            df_raw = chosen_measurement.raw_df.copy()
            if kick_status:
                kicked_dates = dplau.change_value_if_status_not_allowed(df_raw)
                tab1.markdown(f"Plausibilitätstest Status: {len(kicked_dates)} Daten aufgrund des Statuscodes "
                              f"geändert.")
                df_raw = d_p.drop_implausible_measurements(df_raw, kicked_dates)


            if check_gap:
                implausible_dates = dplau.check_gaps(df_raw, min_border=st.session_state["gap_check_min_border"])
                tab1.markdown(f"Plausibilitätstest Lücke: {len(implausible_dates)} unplausible Daten gefunden.")
                df_raw = d_p.drop_implausible_measurements(df_raw, implausible_dates)
            if check_constancy:
                implausible_dates = dplau.check_constancy(df_raw, delta=st.session_state["check_constancy_sum_delta"],
                                                          nr_of_sequential_measurements=st.session_state[
                                                              "check_constancy_nr_sequentiel_measurments"])
                tab1.markdown(f"Plausibilitätstest Konstanz: {len(implausible_dates)} unplausible Daten gefunden.")
                df_raw = d_p.drop_implausible_measurements(df_raw, implausible_dates)
            if check_range:
                implausible_dates = dplau.check_range(df_raw,
                                                      upper_border=st.session_state['range_check_upper_border'],
                                                      lower_border=st.session_state['range_check_lower_border'])
                tab1.markdown(f"Plausibilitätstest Spanne: {len(implausible_dates)} unplausible Daten gefunden.")
                df_raw = d_p.drop_implausible_measurements(df_raw, implausible_dates)

            if check_outlier:
                implausible_dates = dplau.check_outlier(df_raw, r1_perc=st.session_state['r1_check_outlier'],
                                                        r3_perc=st.session_state['r3_check_outlier'])
                tab1.markdown(f"Plausibilitätstest Ausreißer: {len(implausible_dates)} unplausible Daten gefunden.")
                df_raw = d_p.drop_implausible_measurements(df_raw, implausible_dates)
            if check_gradient:
                implausible_dates = dplau.check_gradient(df_raw, delta=st.session_state['gradient_check_delta'])
                tab1.markdown(f"Plausibilitätstest Gradient: {len(implausible_dates)} unplausible Daten gefunden.")
                df_raw = d_p.drop_implausible_measurements(df_raw, implausible_dates)

            if check_noise:
                implausible_dates = dplau.check_noise(df_raw)
                tab1.markdown(f"Plausibilitätstest Rauschen: {len(implausible_dates)} unplausible Daten gefunden.")
                df_raw = d_p.drop_implausible_measurements(df_raw, implausible_dates)
            if check_drift:
                implausible_dates = dplau.check_drift(df_raw,
                                                      chunk_split=st.session_state["check_drift_chunk_split"])
                tab1.markdown(f"Plausibilitätstest Drift: {len(implausible_dates)} unplausible Daten gefunden.")
                df_raw = d_p.drop_implausible_measurements(df_raw, implausible_dates)

            if "selected_fill_method" in st.session_state.keys() and st.session_state["selected_fill_method"] != "Regression":
                df_height_filled = data_filling_fun_dict[selected_fill_method](
                    df_raw) if selected_fill_method else df_raw

            else:
                df_height_filled = df_raw.copy()
            st.session_state["changed_df"] = df_height_filled

            df_nan = df_raw.copy()
            df_nan["value"] = np.nan
            indices_of_nan = pd.isnull(df_raw).any(axis=1).to_numpy().nonzero()[0]
            df_nan.iloc[indices_of_nan] = chosen_measurement.raw_df.iloc[indices_of_nan]
            df_nan = d_p.df_to_datetime(df_nan)
            df_nan = d_p.drop_duplicated_indices(df_nan)

            tab1.write("Messreihe ohne unplausible Daten:")
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
            fig.update_yaxes(ticksuffix=" m")
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Messung",
                              plot_bgcolor="white", margin=dict(t=40, r=1, l=1))

            tab1.plotly_chart(fig, use_container_width=True)

        except KeyError as ke:
            print(ke.args)
            tab1.markdown('<p class="small-font-red">Keine Messreihe ausgewählt</p>', unsafe_allow_html=True)

tab2.markdown('<p class="small-font-red">Händische Datenprüfung ist noch in der Entwicklung!</p>',
              unsafe_allow_html=True)
