import streamlit as st

# Create a Streamlit app
st.set_page_config(page_title="Retail Stock Inventory Analysis", page_icon="✅")

# Define a function for the Home page
def home_page():
    st.title("Welcome to Retail Stock Inventory Analysis")
    st.write("This app allows you to analyze and manage retail stock inventory.")
    
    # Add some styling and layout to the text
    st.markdown("---")  # Add a horizontal line
    st.subheader("Get Started:")
    st.markdown(
        """
        1. **Upload Data:** Navigate to the 'Upload and Analyze' page to upload your retail stock data.
        2. **Analyze Data:** Explore your data, perform analysis, and manage your inventory.
        3. **Identify Outliers:** Visit the 'View Outliers' page to analyze and identify outliers.
        """
    )
    st.markdown("---")

