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

    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Messung",
                      plot_bgcolor="white", margin=dict(t=40, r=0, l=0))

    st.plotly_chart(fig, use_container_width=True)

tab1, tab2 = st.tabs(["Automatische Prüfung", "Händische Prüfung"])
col1, col2 = tab1.columns(2)

if chosen_measurement and not chosen_measurement.status_available:
    kick_status = col1.checkbox('Prüfung anhand Statuscode', disabled=False,
                                help="Kein Statuscode verfügbar.")
else:
    kick_status = col1.checkbox('Prüfung anhand Statuscode')

col1.write("Plausibilitätstests auswählen:")
check_gap = col1.checkbox('Lücke', disabled=False,
                          help="")
if check_gap:
    exp_ = col2.expander("Einstellungen 'Lücke'")
    with exp_:
        st.session_state['gap_check_custom_missing_values'] = st.text_input("Fehlender Wert Indikator")


check_constancy = col1.checkbox('Konstanz', disabled=False,
                                    help=(""))

if check_constancy:
    exp_ = col2.expander("Einstellungen 'Konstanz'")

    st.session_state['check_constancy_nr_sequentiel_measurments'] = \
            st.number_input("Anzahl n der sequenziellen Zeitschritte",
                            min_value=2,
                            max_value=15, value=3, step=1)
    st.session_state['check_constancy_sum_delta'] = \
            st.number_input("Betrag der maximalen Wertänderung in n Zeitschritten",
                            min_value=3.00,
                            max_value=100.00, value=10.0, step=0.01)

check_range = col1.checkbox('Spanne',
                            help="")
if check_range:
    exp_ = col2.expander("Einstellungen 'Spanne'")
    with exp_:
        st.session_state['range_check_lower_border'] = st.number_input("Unterer Grenzwert", value=0.00, step=0.01)
        st.session_state['range_check_upper_border'] = st.number_input("Oberer Grenzwert", value=4.00, step=0.01)

check_outlier = col1.checkbox('Ausreißer', disabled=False, help="")

if check_outlier:
    exp_ = col2.expander("Einstellungen 'Ausreißer'")
    with exp_:
        st.markdown("""
       
        """)

        st.session_state['outlier_method'] = st.selectbox(label="Ausreißer Methode", options=["Standardabweichung-Methode",
                                                                                              "Interquartilbereich-Methode"])
        if st.session_state['outlier_method'] == "Standardabweichung-Methode":
            st.session_state['std_multiplier'] = round(st.number_input("Standardabweichung Multiplikator",
                                                                        min_value=1.0, max_value=50.0,
                                                                        value=3.0,
                                                                        step=0.5) / 100, 2)

        else:
            st.session_state['iqr_multiplier'] = round(st.number_input("Interquartilbereich Multiplikator",
                                                                        min_value=1.0, max_value=50.0,
                                                                        value=1.5,
                                                                        step=0.5) / 100, 2)

check_gradient = col1.checkbox('Gradient', disabled=False, help="")
if check_gradient:
    exp_ = col2.expander("Einstellungen 'Gradient'")
    st.session_state['gradient_check_delta'] = st.number_input("Maximale Wertänderung je Minute",
                                                                   min_value=-100.00,
                                                                   max_value=10.00, value=1.50, step=0.01)

check_noise = col1.checkbox('Rauschen', disabled=False,
                            help="")
if check_noise:
    exp_ = col2.expander("Einstellungen 'Rauschen'")
    with exp_:
        st.write("")

check_drift = col1.checkbox('Drift', disabled=False,
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
                df_gap = dplau.check_gaps(df_raw, custom_missing_values=st.session_state[
                    "gap_check_custom_missing_values"])
                tab1.markdown(f"Plausibilitätstest Lücke: {df_gap['Missing'].sum()} unplausible Daten gefunden.")

            if check_constancy:
                implausible_dates = dplau.check_constancy(df_raw, delta=st.session_state["check_constancy_sum_delta"],
                                                          nr_of_sequential_measurements=st.session_state[
                                                              "check_constancy_nr_sequentiel_measurments"])
                tab1.markdown(f"Plausibilitätstest Konstanz: {len(implausible_dates)} unplausible Daten gefunden.")
                df_raw = d_p.drop_implausible_measurements(df_raw, implausible_dates)
            if check_range:
                df_range = dplau.check_range(df_raw,
                                                      upper_border=st.session_state['range_check_upper_border'],
                                                      lower_border=st.session_state['range_check_lower_border'])
                tab1.markdown(f"Plausibilitätstest Spanne: {df_range['Range Error'].sum()} unplausible Daten gefunden.")

            if check_outlier:
                if st.session_state['outlier_method'] == "Standardabweichung-Methode":
                    df_outlier = dplau.check_outlier(df_raw, std_multiplier=st.session_state['std_multiplier'])
                else:
                    df_outlier = dplau.check_outlier(df_raw, iqr_multiplier=st.session_state['iqr_multiplier'])
                tab1.markdown(f"Plausibilitätstest Ausreißer: {df_outlier['Outlier'].sum()} unplausible Daten gefunden.")
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

            if "selected_fill_method" in st.session_state.keys() and st.session_state[
                "selected_fill_method"] != "Regression":
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


            if hasattr(chosen_measurement, "days_changed_dict"):
                del chosen_measurement.__dict__["days_changed_dict"]
            st.session_state["validation_check_iteration_nr"] = 0
            chosen_measurement.validated_df = st.session_state["changed_df"]

            # tab1.success(f'Die validierten Daten wurden zur Messreihe {chosen_measurement.name} hinzugefügt.')


        except KeyError as ke:
            print(ke.args)
            tab1.markdown('<p class="small-font-red">Keine Messreihe ausgewählt</p>', unsafe_allow_html=True)

tab2.markdown('<p class="small-font-red">Händische Datenprüfung ist noch in der Entwicklung!</p>',
              unsafe_allow_html=True)
