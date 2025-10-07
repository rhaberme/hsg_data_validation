import streamlit as st

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
st.title("Impressum")
st.write("""Hochschulgruppe Simulation vertreten durch: Ralf Habermehl \n Paul-Ehrlich-Straße 14 \n Gebäude: 14, Raum: 320 \n 67663 Kaiserslautern

Kontakt: E-Mail: info@hsgsim.org""")