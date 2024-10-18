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
