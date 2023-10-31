import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import datetime
import plotly.figure_factory as ff
import streamlit as st 


def outliers_check(data):
            fig, axs = plt.subplots(4,figsize=(5,12))
            X = data[['Temperature','Fuel_Price','CPI','Unemployment']]
            for i,column in enumerate(X):
                sns.boxplot(data[column], ax=axs[i])
            st.pyplot(fig)
def outlier_page():
    st.header('Upload Dataset to check for Outliers')
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader('Heres the uploaded *dataset*')
        st.table(data.head(20))
        outliers_check(data)
    