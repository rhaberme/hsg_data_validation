from Measurement import Measurement
import help_functions as s_f
import data_processing as d_p
import data_filling as d_f
import pandas as pd
import streamlit as st
import pathlib
from pathlib import Path
import os

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

.st-key-continue_to_plausibility button {
    background-color: #2e7d32 !important;
    border-color: #2e7d32 !important;
    color: #ffffff !important;
}
.st-key-continue_to_plausibility button:hover,
.st-key-continue_to_plausibility button:focus,
.st-key-continue_to_plausibility button:active {
    background-color: #1b5e20 !important;
    border-color: #1b5e20 !important;
    color: #ffffff !important;
}
</style>""", unsafe_allow_html=True)

# Main area content
st.title("Upload data")
st.write("Here you can upload the time series for plausibility tests.")

# Helper function to get available sample datasets
def get_sample_datasets():
    """Discover CSV files in the data folder."""
    data_folder = pathlib.Path(__file__).parent.parent.parent.resolve() / "data"
    
    if not data_folder.exists():
        return []
    
    sample_files = sorted([
        f for f in data_folder.iterdir()
        if f.is_file() 
        and f.suffix.lower() in ('.csv')
        and f.name != '.gitkeep'  # Explicitly exclude .gitkeep
    ])
    
    return sample_files

# Load sample datasets
sample_datasets = get_sample_datasets()
sample_datasets_dict = {f.name: f for f in sample_datasets}

name = st.text_input("Time series name:", placeholder="Enter time series name. Defaults to filename and column.")

# Tabs for file upload vs. sample data
tab1, tab2 = st.tabs(["Upload from file", "Use sample dataset"])

file_to_process = None
file_source = None  # Track whether it's "upload" or "sample"

# TAB 1: Upload from file
with tab1:
    st.session_state["measurement_uploader"] = st.file_uploader('Load file', type=['ftr', 'csv', 'txt'])
    
    CUSTOM_SEPARATOR = "__custom__"
    separator_options = {
        "Auto-detect": None,
        "Comma  ,": ",",
        "Semicolon  ;": ";",
        "Tab  \\t": "\t",
        "Space": " ",
        "Custom...": CUSTOM_SEPARATOR,
    }
    
    datetime_in_two_col = st.checkbox("Date and time provided in separate columns?", value=False)
    separator_choice = st.selectbox("Separator", separator_options, index=0)
    selected_separator = separator_options[separator_choice]
    
    if selected_separator == CUSTOM_SEPARATOR:
        selected_separator = st.text_input(
            "Custom separator", max_chars=8,
            help="Entered literally, so a single character like | is what you want here. "
                 "Use the Tab option above for tab separated files.")
        if selected_separator == "":
            st.info("No custom separator entered yet - detecting it automatically.")
            selected_separator = None

    if st.button("Import dataset"):
        if st.session_state["measurement_uploader"]:
            st.session_state["file_to_process"] = st.session_state["measurement_uploader"]
            st.session_state["file_source"] = "upload"
            st.success("File imported successfully. Please configure import settings below.")
        else:
            st.warning("Please upload a file first before importing.")
        
    

# TAB 2: Sample dataset
with tab2:
    if sample_datasets:
        sample_file = st.selectbox("Select a sample dataset:", list(sample_datasets_dict.keys()))
        if st.button("Load sample dataset"):
            st.session_state["file_to_process"] = sample_datasets_dict[sample_file]
            st.session_state["file_source"] = "sample"
            st.session_state["selected_separator"] = None  # Auto-detect for samples
            st.session_state["datetime_in_two_col"] = False  # Will be detected
            st.success("sample dataset loaded successfully. Please configure import settings below.")
    else:
        st.info("No sample datasets available in the data folder.")

# Initialize session state
if "column_names" not in st.session_state.keys():
    st.session_state["column_names"] = []
    
if "measurement_dict" not in st.session_state:
    st.session_state["measurement_dict"] = {}
    
if "file_to_process" not in st.session_state:
    st.session_state["file_to_process"] = None
    
if "file_source" not in st.session_state:
    st.session_state["file_source"] = None

# ============================================================================
# UNIFIED IMPORT/PROCESSING LOGIC (runs for both upload and sample)
# ============================================================================

if st.session_state["file_to_process"] is not None:
    file_to_process = st.session_state["file_to_process"]
    file_source = st.session_state["file_source"]
    
    st.divider()
    st.subheader("Configure import settings")
    
    col1, col2 = st.columns(2)
    
    status_available = st.checkbox("Status available?", value=False)
    
    # Save file if it's an upload; if it's a sample, use the path directly
    if file_source == "upload":
        current_directory = str(pathlib.Path(__file__).parent.resolve()).rstrip("pages")
        s_f.save_uploaded_file(uploadedfile=file_to_process,
                               filename=file_to_process.name,
                               current_directory=current_directory)
        filepath = current_directory + "tempDir/" + file_to_process.name
    else:  # file_source == "sample"
        filepath = str(file_to_process)
    
    # Detect separator
    sep = d_p.detect_separator(filepath) if selected_separator is None else selected_separator

    # Read column names
    try:
        st.session_state["column_names"] = pd.read_csv(filepath, on_bad_lines='skip', sep=sep).columns
    except (ValueError, pd.errors.ParserError) as parse_error:
        st.session_state["column_names"] = []
        st.error(f"Could not read the file with separator {sep!r}: {parse_error}")

    if len(st.session_state["column_names"]) == 1:
        st.warning(f"Separator {sep!r} splits the file into a single column. "
                   "If that is not intended, pick a different separator above.")

    # Column selection
    label_value = st.selectbox("Name of the measurement-column", st.session_state["column_names"],
                               index=st.session_state["column_names"].get_loc("value")
                               if "value" in st.session_state["column_names"] else 1)

    if datetime_in_two_col:
        label_date = st.selectbox("Name of the date-column", st.session_state["column_names"],
                                  index=st.session_state["column_names"].get_loc("date")
                                  if "date" in st.session_state["column_names"] else 0)
        label_time = st.selectbox("Name of the time-column", st.session_state["column_names"],
                                  index=st.session_state["column_names"].get_loc("time")
                                  if "time" in st.session_state["column_names"] else 0)
        label_date_time = [label_date, label_time]
    else:
        label_date_time_string = st.selectbox("Name of the datetime-column", st.session_state["column_names"])
        label_date_time = [label_date_time_string]

    # Detect timesteps
    timesteps = []
    if label_date_time is not None:
        try:
            raw_df = d_p.create_df_data_from_csv(filepath, date_time_col_name_s=label_date_time,
                                                 value_col_name=label_value,
                                                 ambiguous=False,
                                                 status_available=status_available,
                                                 sep=sep)
            timesteps = list(map(int, d_p.check_timesteps(raw_df)["checktimes"].tolist()))
        except KeyError:
            pass

    # Sampling frequency and resampling
    sampling_freq = st.selectbox("Sampling frequency [s]", timesteps, index=0, accept_new_options=True)
    fill_resampling = st.selectbox("Resampling fill method (Leave empty to fill after plausibility tests)", 
                                   d_f.data_filling_fun_dict, index=None)

    if sampling_freq is not None:
        sampling_freq = int(sampling_freq)
    if sampling_freq is not None and not (sampling_freq in timesteps or any(map(lambda ts: (ts/sampling_freq) %1 == 0 or (sampling_freq/ts) %1 == 0, timesteps))):
        st.warning("Sampling frequency is not yet in the time series or an integer multiple/divisor of any of them. This might cause strange results.")

    # Status column
    if status_available:
        label_status = st.selectbox("Name of the status-column", st.session_state["column_names"])
        accepted_status = st.text_input("Accepted status")
    else:
        label_status = None
        accepted_status = None

    # Initialize measurement dict
    if "measurement_dict" not in st.session_state:
        st.session_state["measurement_dict"] = {}

    # Add data button
    measurement_submit_button = st.button(label='Add data', type="primary")
    if measurement_submit_button:
        with st.spinner('Time series is added...'):
            # If no name is given, use filename + column name
            if name == "":
                display_name = file_to_process.name if file_source == "upload" else Path(file_to_process).name
                name = display_name + " " + label_value

            # Load data
            measurement_instance = Measurement(filepath=filepath, name=name,
                                               measurement_type=None, label_value=label_value,
                                               label_date_time=label_date_time, sampling_freq=sampling_freq,
                                               fill_resampling=fill_resampling,
                                               status_available=status_available, label_status=label_status,
                                               accepted_status=accepted_status, sep=sep, drop_duplicates=True)
            st.session_state["measurement_dict"][measurement_instance.name] = measurement_instance
        st.success(f'The time series {name} has been successfully added.')

# ============================================================================
# Display added measurements and continue button
# ============================================================================

if st.session_state["measurement_dict"]:
    st.divider()
    st.subheader("Added time series")
    for meas_name in st.session_state["measurement_dict"].keys():
        st.write(f"✓ {meas_name}")
    
    if st.button("Continue to Plausibility Tests", type="primary", key="continue_to_plausibility"):
        st.switch_page("pages/3_Plausibility_tests.py")