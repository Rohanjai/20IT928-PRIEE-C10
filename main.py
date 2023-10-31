import streamlit as st
from app import *
from home import *
# Create a Streamlit app
# st.title("*RETAIL STOCK STORE INVENTORY ANALYSIS*")
st.title("RETAIL STOCK STORE INVENTORY ANALYSIS")
st.markdown("---")
# Define different pages
pages = {
    "Home": "Home",
    "Upload and Analyze": "Upload and Analyze",
    "View Outliers": "View Outliers",
}

# Create a sidebar with links to navigate to different pages
selected_page = st.sidebar.radio("Select a Page", list(pages.keys()))

# Page for the "Home" link
if selected_page == "Home":
    home_page()
    

# Page for the "Upload and Analyze" link
if selected_page == "Upload and Analyze":
    st.title("Upload and Analyze Dataset")
    # Add code for the "Upload and Analyze" page
    Analyze_Page()

# Page for the "View Outliers" link
if selected_page == "View Outliers":
    st.title("View Outliers")
    # Add code for the "View Outliers" page

# Additional content in your app
