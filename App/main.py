import streamlit as st
from app import *
from home import *
from Forecasting import *
from Predictions import *


st.title("RETAIL STOCK STORE INVENTORY ANALYSIS")
st.markdown("---")

pages = {
    "Home": "Home",
    "Upload and Analyze": "Upload and Analyze",
    "Predictions Page": "Predictions Page",
    "Forecasting Page": "Forecasting Page",

}
selected_page = st.sidebar.radio("Select a Page", list(pages.keys()))
if selected_page == "Home":
    home_page()
elif selected_page == "Upload and Analyze":
    # st.title("Upload and Analyze Dataset")
    Analyze_Page()
elif selected_page == "Predictions Page":
    # st.title("Predictions Page")
    Predictions_Page()
elif selected_page == "Forecasting Page":
    # st.subheader("Forecasting Page")
    forecasting_page()



