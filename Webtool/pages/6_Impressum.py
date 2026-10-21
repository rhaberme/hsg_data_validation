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
st.write("""
    Hochschulgruppe Simulation [https://hsgsim.org/] represented by: Ralf Habermehl
    
    Paul-Ehrlich-Straße 14
    
    Building: 14
    
    Room: 320
    
    67663 Kaiserslautern
    
    Contact: E-Mail: info@hsgsim.org
    """)
