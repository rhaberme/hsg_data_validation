from Measurement import Measurement
import help_functions as s_f
import pandas as pd
import streamlit as st
import pathlib

st.set_page_config(
    page_title="HSG Sim Datentool",
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
</style>""", unsafe_allow_html=True)


st.write("""
# Messstelle hinzufügen
## MW-Entlastungsanlage
""")

col1, col2 = st.columns(2)

lat = None
lon = None



st.write("""
## Messreihe hinzufügen
""")



measurment_number = st.text_input("Messstellen-Nr.")
name = st.text_input("Name der Messreihe")

from_file_exp = st.expander("Messreihe von Datei laden", expanded=True)



status_available = from_file_exp.checkbox("Status verfügbar?")
datetime_in_two_col = from_file_exp.checkbox("Datum und Zeit in zwei Spalten aufgeteilt?",
                                             value=True)


# with st.form(key='add_measurement_form'):
st.session_state["measurement_uploader"] = from_file_exp.file_uploader('Messreihe hochladen', type=['ftr', 'csv', 'txt'])
if "column_names" not in st.session_state.keys():
    st.session_state["column_names"] = []

if st.session_state["measurement_uploader"]:
    current_directory = str(pathlib.Path(__file__).parent.resolve()).rstrip("pages")
    s_f.save_uploaded_file(uploadedfile=st.session_state["measurement_uploader"],
                           filename=st.session_state["measurement_uploader"].name,
                           current_directory=current_directory
                           )
    filepath = current_directory + "tempDir/" + st.session_state[
        "measurement_uploader"].name
    column_names = None
    for sep_ in [",", ";", " "]:
        current_column_names = pd.read_csv(filepath, on_bad_lines='skip', sep=sep_).columns
        if column_names is not None:
            if len(current_column_names) > len(column_names):
                column_names = current_column_names
                sep = sep_

        else:
            column_names = current_column_names
            sep = sep_


    st.session_state["column_names"] = column_names


label_value = from_file_exp.selectbox("Label der Messwert-Spalte", st.session_state["column_names"],
                                      index=st.session_state["column_names"].get_loc("value")
                                      if "value" in st.session_state["column_names"] else 0)

if datetime_in_two_col:
    label_date = from_file_exp.selectbox("Label der Datum-Spalte", st.session_state["column_names"],
                                         index=st.session_state["column_names"].get_loc("date")
                                         if "date" in st.session_state["column_names"] else 0
                                         )
    label_time = from_file_exp.selectbox("Label der Zeit-Spalte", st.session_state["column_names"],
                                         index=st.session_state["column_names"].get_loc("time")
                                         if "time" in st.session_state["column_names"] else 0
                                         )
    label_date_time = [label_date, label_time]
else:
    label_date_time_string = from_file_exp.selectbox("Label der Datum-Spalte", st.session_state["column_names"])
    label_date_time = [label_date_time_string]

if status_available:
    label_status = from_file_exp.selectbox("Label der Status-Spalte", st.session_state["column_names"])
    accepted_status = from_file_exp.text_input("Akzeptierter Status")
else:
    label_status = None
    accepted_status = None

if "measurment_dict" not in st.session_state:
    st.session_state["measurement_dict"] = {}
measurement_submit_button = from_file_exp.button(label='Hinzufügen')
if measurement_submit_button:
    with st.spinner('Messreihe wird hinzugefügt...'):

        measurement_instance = Measurement(filepath=filepath, name=name,
                                           measurement_type=None, label_value=label_value,
                                           label_date_time=label_date_time, status_available=status_available,
                                           label_status=label_status, accepted_status=accepted_status, sep=sep,
                                           drop_duplicates=True)
        st.session_state["measurement_dict"][measurement_instance.name] = measurement_instance
    st.success(f'Die Messreihe {name} wurde erfolgreich hinzugefügt.')
