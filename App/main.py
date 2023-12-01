import streamlit as st
from app import *
from home import *
from Forecasting import *
from Predictions import *


st.title("RETAIL STOCK STORE INVENTORY ANALYSIS")
st.markdown("---")

pages = {
    "Home": "Home",
    "Analysis Page": "Analysis Page",
    "Sales Predictions Page": "SalesPredictions Page",
    "Demand Forecasting Page": "Demand Forecasting Page",

}
selected_page = st.sidebar.radio("Select a Page", list(pages.keys()))
if selected_page == "Home":
    home_page()
elif selected_page == "Analysis Page":
    # st.title("Upload and Analyze Dataset")
    Analyze_Page()
elif selected_page == "Sales Predictions Page":
    # st.title("Predictions Page")
    Predictions_Page()
elif selected_page == "Demand Forecasting Page":
    # st.subheader("Forecasting Page")
    forecasting_page()



