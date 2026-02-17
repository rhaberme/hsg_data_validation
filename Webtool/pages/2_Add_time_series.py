from Measurement import Measurement
import help_functions as s_f
import data_processing as d_p
import data_filling as d_f
import pandas as pd
import streamlit as st
import pathlib

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
</style>""", unsafe_allow_html=True)

# Main area content
st.title("Upload data")
st.write("Here you can upload the time series for plausibility tests.")

col1, col2 = st.columns(2)

lat = None
lon = None


name = st.text_input("Time series name:", placeholder="Enter time series name. Defaults to filename and column.")

from_file_exp = st.expander("Load time series from file", expanded=True)



status_available = from_file_exp.checkbox("Status available?")
datetime_in_two_col = from_file_exp.checkbox("Date and time provided in seperate columns?",
                                             value=False)


# with st.form(key='add_measurement_form'):
st.session_state["measurement_uploader"] = from_file_exp.file_uploader('Load file', type=['ftr', 'csv', 'txt'])
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


label_value = from_file_exp.selectbox("Name of the measurement-column", st.session_state["column_names"],
                                      index=st.session_state["column_names"].get_loc("value")
                                      if "value" in st.session_state["column_names"] else 1)

if datetime_in_two_col:
    label_date = from_file_exp.selectbox("Name of the date-column", st.session_state["column_names"],
                                         index=st.session_state["column_names"].get_loc("date")
                                         if "date" in st.session_state["column_names"] else 0
                                         )
    label_time = from_file_exp.selectbox("Name of the time-column", st.session_state["column_names"],
                                         index=st.session_state["column_names"].get_loc("time")
                                         if "time" in st.session_state["column_names"] else 0
                                         )
    label_date_time = [label_date, label_time]
else:
    label_date_time_string = from_file_exp.selectbox("Name of the timedate-column", st.session_state["column_names"])
    label_date_time = [label_date_time_string]


timesteps = []
if st.session_state["measurement_uploader"] and label_date_time is not None:
    try:
        raw_df = d_p.create_df_data_from_csv(filepath, date_time_col_name_s=label_date_time,
                                             value_col_name=label_value,
                                             ambiguous=False,
                                             status_available=status_available,
                                             sep=sep
                                             )
        timesteps = list(map(int, d_p.check_timesteps(raw_df)["checktimes"].tolist()))
    except KeyError:
        pass

sampling_freq = from_file_exp.selectbox("Sampling frequency [s]", timesteps, index=0, accept_new_options=True)
fill_resampling = from_file_exp.selectbox("Resampling fill method (Leave empty to fill after plausibility tests)", d_f.data_filling_fun_dict, index=None)

if sampling_freq is not None:
    sampling_freq = int(sampling_freq)
if sampling_freq is not None and not (sampling_freq in timesteps or any(map(lambda ts: (ts/sampling_freq) %1 == 0 or (sampling_freq/ts) %1 == 0, timesteps))):
    st.warning("Sampling frequency is not yet in the time series or an integer multiple/divisor of any of them. This might cause strange results.")

if status_available:
    label_status = from_file_exp.selectbox("Name of the status-column", st.session_state["column_names"])
    accepted_status = from_file_exp.text_input("Accepted status")
else:
    label_status = None
    accepted_status = None

if "measurement_dict" not in st.session_state:
    st.session_state["measurement_dict"] = {}
measurement_submit_button = from_file_exp.button(label='Add data')
if measurement_submit_button:
    with st.spinner('Time series is added...'):
        # If no name is given, use filename + column name
        if name == "":
            name = st.session_state["measurement_uploader"].name + " " + label_value

        # Load data
        measurement_instance = Measurement(filepath=filepath, name=name,
                                           measurement_type=None, label_value=label_value,
                                           label_date_time=label_date_time, sampling_freq=sampling_freq,
                                           fill_resampling=fill_resampling,
                                           status_available=status_available, label_status=label_status,
                                           accepted_status=accepted_status, sep=sep, drop_duplicates=True)
        st.session_state["measurement_dict"][measurement_instance.name] = measurement_instance
    st.success(f'The time series {name} has been successfully added.')
