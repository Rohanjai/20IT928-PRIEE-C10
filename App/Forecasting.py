import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import datetime
import plotly.figure_factory as ff
import streamlit as st 
import tensorflow as tf

test = pd.read_csv('../Data/test.csv')
def prediction_lstm(X_test):
    model = tf.keras.models.load_model('../models/lstm_model.h5')
    lstm_test_pred = model.predict(X_test)
    lstm_prediction = pd.DataFrame(test['ID'], columns=['ID'])
    lstm_prediction['item_cnt_month'] = lstm_test_pred.clip(0., 20.)
    # lstm_prediction.to_csv('lstm_predictions.csv', index=False)
    st.table(lstm_prediction.head(10))


def prediction_mlp(X_test):
    model = tf.keras.models.load_model('../models/mlp_model.h5')
    # encoder = tf.keras.models.load_model('encoder_decoder.h5')
    mlp_test_pred = model.predict(X_test)
    mlp_prediction = pd.DataFrame(test['ID'], columns=['ID'])
    mlp_prediction['item_cnt_month'] = mlp_test_pred.clip(0., 20.)
    # mlp_prediction.to_csv('mlp_predictions.csv', index=False)
    st.table(mlp_prediction.head(10))


def preprocessing(X_test):
    X_test = X_test.drop(['Unnamed: 0'], axis=1)
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    return X_test_reshaped

def forecasting_page():
    st.subheader('Upload Dataset to check for Forecasting')
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader('Heres the uploaded *dataset*')
        st.table(data.head(5))
    #Drop down menu for forecasting
    st.subheader('Select the model for forecasting')
    option = st.selectbox('Select the model',('None','MLP','LSTM'))

    if option == 'None':
        #write select a model in center
        st.subheader('Select a model to start forecasting')



    elif option == 'LSTM':
        st.write('You selected', option)
        if st.button('Start Forecasting',key='1'):
            
            test_data = preprocessing(data)
            with st.spinner("Calculating model predictions in the background... Please wait."):
                prediction_lstm(test_data)
            
    elif option == 'MLP':
        st.write('You selected', option)
        if st.button('Start Forecasting',key='2'):
            test_data = preprocessing(data)
            # st.info("Calculating model predictions in the background... Please wait.")
            with st.spinner("Calculating model predictions in the background... Please wait."):
                prediction_mlp(test_data)
            
    