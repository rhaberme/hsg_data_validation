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

st.set_page_config(
    page_title="HSG Sim Datentool",
    page_icon="HSG",
    layout="centered"
    # initial_sidebar_state="collapsed",
)

hide_menu = """
<style>
// #MainMenu {visibility:hidden;}
footer{visibility:hidden;}
</style>
"""

st.markdown(hide_menu, unsafe_allow_html=True)

st.write("""
# Startseite
## HSG Sim Datentool
""")

st.write("")



