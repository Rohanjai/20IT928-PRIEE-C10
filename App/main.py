import streamlit as st
from app import *
from home import *
from Forecasting import *

st.title("RETAIL STOCK STORE INVENTORY ANALYSIS")
st.markdown("---")

pages = {
    "Home": "Home",
    "Upload and Analyze": "Upload and Analyze",
    "Forecasting Page": "Forecasting Page",
}
selected_page = st.sidebar.radio("Select a Page", list(pages.keys()))
if selected_page == "Home":
    home_page()
elif selected_page == "Upload and Analyze":
    st.title("Upload and Analyze Dataset")
    Analyze_Page()
elif selected_page == "Forecasting Page":
    st.title("Forecasting Page")
    forecasting_page()


