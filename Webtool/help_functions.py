import streamlit as st
import os


def save_uploaded_file(uploadedfile, filename, current_directory=None):
    if current_directory:
        target_directory = os.path.join(current_directory, "tempDir")
    else:
        target_directory = os.path.join("../../Desktop/liplau-master", "tempDir")

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    with open(os.path.join(target_directory, filename), "wb") as f:
        f.write(uploadedfile.getbuffer())
