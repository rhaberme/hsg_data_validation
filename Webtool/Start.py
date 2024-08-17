import streamlit as st

st.set_page_config(
    page_title="CSO Data-Tool",
    page_icon="🌊",
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



