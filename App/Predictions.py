import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
def Predictions_Page():
    rfr = pickle.load(open('../models/model_rfr.pkl', 'rb'))
    knn = pickle.load(open('../models/model_knn.pkl', 'rb'))
    lr = pickle.load(open('../models/model_reg.pkl', 'rb'))
    gbr = pickle.load(open('../models/model_gbr.pkl', 'rb'))

    st.subheader('Enter the features to get the predictions',divider=True)
    current_date = datetime.date.today()
    store_number = st.number_input("Store No", min_value=1)
    fuel_price = st.number_input("Fuel Price")
    cpi = st.number_input("CPI")
    unemployment = st.number_input("Unemployment")
    day = st.slider("Day", 1, 31, current_date.day)
    month = st.slider("Month", 1, 12, current_date.month)
    year = st.slider("Year", 2000, current_date.year, current_date.year)
    features = np.array([store_number,fuel_price,cpi,unemployment,day,month,year])
    # for i in ['Store No','Fuel_Price','CPI','Unemployment','Day','Month','Year']:
        # st.markdown(i)
    st.divider()
    st.subheader('Available Models') 
    tab1,tab2,tab3,tab4 = st.tabs(['Linear Regression','Random Forest Regressor','Knn Regressor','Gradient Boosting Regressor'])
    
    with tab1:
        if st.button('Predict',key='1'):
            prediction = lr.predict(features.reshape(1,-1))
            st.success('The predicted weekly sales is {}'.format(prediction))
    
    with tab2:
        if st.button('Predict',key='2'):
            prediction = rfr.predict(features.reshape(1,-1))
            st.success('The predicted weekly sales is {}'.format(prediction))

    with tab3:
        if st.button('Predict',key='3'):
            prediction = knn.predict(features.reshape(1,-1))
            st.success('The predicted weekly sales is {}'.format(prediction))
    with tab4:
        if st.button('Predict',key='4'):
            prediction = gbr.predict(features.reshape(1,-1))
            st.success('The predicted weekly sales is {}'.format(prediction))