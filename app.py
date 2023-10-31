# Import necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import datetime
import plotly.figure_factory as ff
import streamlit as st 
# Load dataset

st.title("*RETAIL STOCK STORE INVENTORY ANALYSIS*")

uploaded_file = st.file_uploader("*Upload the dataset for analysis*")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('Heres the uploaded *dataset*')
    st.write(data)
    # Convert date to datetime format and show dataset information
    data['Date'] =  pd.to_datetime(data['Date'])
    data.info()
    # checking for missing values
    data.isnull().sum()
    # Splitting Date and create new columns (Day, Month, and Year)
    data['Month']= pd.DatetimeIndex(data['Date']).month
    data['Day'] = pd.DatetimeIndex(data['Date']).day
    data['Year'] = pd.DatetimeIndex(data['Date']).year

    st.sidebar.title('*App Navigation*')

    section = st.sidebar.radio('Go to:', ('Section 1', 'Section 2', 'Section 3'))
    plt.figure(figsize=(50,50))
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Sum Weekly_Sales for each store, then sortded by total sales
    total_sales_for_each_store = pd.DataFrame(data.groupby('Store')['Weekly_Sales'].sum().sort_values()) 
    total_sales_for_each_store_array = np.array(total_sales_for_each_store) # convert to array

    # Assigning a specific color for the stores have the lowest and highest sales
    clrs = ['lightsteelblue' if ((x < max(total_sales_for_each_store_array)) and (x > min(total_sales_for_each_store_array))) else 'midnightblue' for x in total_sales_for_each_store_array]


    ax = total_sales_for_each_store.plot(kind='bar',color=clrs)

    # store have minimum sales
    p = ax.patches[0]
    # print(type(p.get_height()))
    ax.annotate("The store has minimum sales is 33 with {0:.2f} $".format((p.get_height())), xy=(p.get_x(), p.get_height()), xycoords='data',
                xytext=(0.17, 0.32), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                horizontalalignment='center', verticalalignment='center')


    # store have maximum sales 
    p = ax.patches[44]
    ax.annotate("The store has maximum sales is 20 with {0:.2f} $".format((p.get_height())), xy=(p.get_x(), p.get_height()), xycoords='data',
                xytext=(0.82, 0.98), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                horizontalalignment='center', verticalalignment='center')


    # plot properties
    plt.xticks(rotation=0)
    plt.ticklabel_format(useOffset=False, style='plain', axis='y')
    plt.title('Total sales for each store')
    plt.xlabel('Store')
    plt.ylabel('Total Sales')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Which store has maximum standard deviation
    st.subheader('Which store has maximum standard deviation')
    data_std = pd.DataFrame(data.groupby('Store')['Weekly_Sales'].std().sort_values(ascending=False))
    st.write("The store has maximum standard deviation is "+str(data_std.head(1).index[0])+" with {0:.0f} $".format(data_std.head(1).Weekly_Sales[data_std.head(1).index[0]]))
    # Distribution of store has maximum standard deviation
    st.subheader('Distribution of store has maximum standard deviation')
    plt.figure(figsize=(15,7))
    a=sns.distplot(data[data['Store'] == data_std.head(1).index[0]]['Weekly_Sales'])
    plt.title('The Sales Distribution of Store #'+ str(data_std.head(1).index[0]))
    st.pyplot()




    plt.figure(figsize=(15,7))

    # Sales for third quarterly in 2012
    Q3 = data[(data['Date'] > '2012-07-01') & (data['Date'] < '2012-09-30')].groupby('Store')['Weekly_Sales'].sum()

    # Sales for second quarterly in 2012
    Q2 = data[(data['Date'] > '2012-04-01') & (data['Date'] < '2012-06-30')].groupby('Store')['Weekly_Sales'].sum()

    # Plotting the difference between sales for second and third quarterly
    # Q2.plot(ax=Q3.plot('bar',legend=True),kind='bar',color='r',alpha=0.2,legend=True);
    # plt.legend(["Q3' 2012", "Q2' 2012"]);
    st.subheader('The difference between sales for second and third quarterly')
    fig, ax = plt.subplots()

    # Plot the sales for the second quarterly
    Q2.plot(kind='bar', color='b', alpha=0.2, ax=ax)

    # Plot the sales for the third quarterly on the same axis
    Q3.plot(kind='bar', color='r', alpha=0.2, ax=ax)

    # Add a legend
    plt.legend(["Q2' 2012", "Q3' 2012"])

    st.pyplot()


    st.write('Store have good quarterly growth rate in Q3’2012 is Store '+str(Q3.idxmax())+' With '+str(Q3.max())+' $')

    
    
    def plot_line(df,holiday_dates,holiday_label):
        fig, ax = plt.subplots(figsize = (15,5))  
        ax.plot(df['Date'],df['Weekly_Sales'],label=holiday_label)
        
        for day in holiday_dates:
            day = datetime.strptime(day, '%d-%m-%Y')
            plt.axvline(x=day, linestyle='--', c='r')
        

        plt.title(holiday_label)
        x_dates = df['Date'].dt.strftime('%Y-%m-%d').sort_values().unique()
        xfmt = dates.DateFormatter('%d-%m-%y')
        ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.set_major_locator(dates.DayLocator(1))
        plt.gcf().autofmt_xdate(rotation=90)
        st.pyplot()


    total_sales = data.groupby('Date')['Weekly_Sales'].sum().reset_index()
    Super_Bowl =['12-2-2010', '11-2-2011', '10-2-2012']
    Labour_Day =  ['10-9-2010', '9-9-2011', '7-9-2012']
    Thanksgiving =  ['26-11-2010', '25-11-2011', '23-11-2012']
    Christmas = ['31-12-2010', '30-12-2011', '28-12-2012']
    st.header('*Sales in holidays*')
    tab1 ,tab2 , tab3,tab4 = st.tabs(['Super Bowl','Thanksgiving','Labour Day','Christmas'])
    with tab1:
        st.line_chart(total_sales,Super_Bowl,'Super Bowl')
    with tab2:
        plot_line(total_sales,Labour_Day,'Labour Day')
    with tab3:
        plot_line(total_sales,Thanksgiving,'Thanksgiving')
    with tab4:
        plot_line(total_sales,Christmas,'Christmas')
    
    
    
    # data.loc[data.Date.isin(Super_Bowl)]



    # Yearly Sales in holidays
    st.header('*Yearly Sales in holidays*')
    Super_Bowl_df = pd.DataFrame(data.loc[data.Date.isin(Super_Bowl)].groupby('Year')['Weekly_Sales'].sum())
    Thanksgiving_df = pd.DataFrame(data.loc[data.Date.isin(Thanksgiving)].groupby('Year')['Weekly_Sales'].sum())
    Labour_Day_df = pd.DataFrame(data.loc[data.Date.isin(Labour_Day)].groupby('Year')['Weekly_Sales'].sum())
    Christmas_df = pd.DataFrame(data.loc[data.Date.isin(Christmas)].groupby('Year')['Weekly_Sales'].sum())
    tab1 ,tab2 , tab3,tab4 = st.tabs(['Super Bowl','Thanksgiving','Labour Day','Christmas'])
    with tab1:
        st.subheader('Super Bowl',divider='blue')
        st.text('Yearly Sales in Super Bowl holiday')
        # Super_Bowl_df.plot(kind='bar',legend=False,title='Yearly Sales in Super Bowl holiday') 
        st.bar_chart(Super_Bowl_df)
    with tab2:
        st.subheader('Thanksgiving',divider='blue')
        st.text('Yearly Sales in Thanksgiving holiday')
        # Thanksgiving_df.plot(kind='bar',legend=False,title='Yearly Sales in Thanksgiving holiday') 
        st.bar_chart(Thanksgiving_df)
    with tab3:
        st.subheader('Labour Day',divider='blue')
        # Labour_Day_df.plot(kind='bar',legend=False,title='Yearly Sales in Labour_Day holiday')
        st.text('Yearly Sales in Labour Day holiday')
        st.bar_chart(Labour_Day_df)
    with tab4:
        st.subheader('Christmas',divider='blue')
        # Christmas_df.plot(kind='bar',legend=False,title='Yearly Sales in Christmas holiday')
        st.text('Yearly Sales in Christmas holiday')
        st.bar_chart(Christmas_df)



    st.header('*Monthly view of sales for each years*')
    
    # Monthly view of sales for each years
    tab1 ,tab2 , tab3 = st.tabs(['2010','2011','2012'])
    with tab1:
        st.subheader('2010')
        plt.scatter(data[data.Year==2010]["Month"],data[data.Year==2010]["Weekly_Sales"])
        plt.xlabel("months")
        plt.ylabel("Weekly Sales")
        plt.title("Monthly view of sales in 2010")
        st.pyplot()
    with tab2:
        st.subheader('2011')
        plt.scatter(data[data.Year==2011]["Month"],data[data.Year==2011]["Weekly_Sales"])
        plt.xlabel("months")
        plt.ylabel("Weekly Sales")
        plt.title("Monthly view of sales in 2011")
        st.pyplot()
    with tab3:
        st.subheader('2012')
        plt.scatter(data[data.Year==2012]["Month"],data[data.Year==2012]["Weekly_Sales"])
        plt.xlabel("Months")
        plt.ylabel("Weekly Sales")
        plt.title("Monthly view of sales in 2012")
        st.pyplot()




    # Monthly view of sales for all years
    st.header('*Monthly view of sales for all years*')
    plt.figure(figsize=(10,6))
    # st.bar_chart(data["Month"],data["Weekly_Sales"],x="Month",y="Weekly_Sales")
    # plt.xlabel("months")
    # plt.ylabel("Weekly Sales")
    # plt.title("Monthly view of sales")
    monthly_data = data.groupby("Month")[["Weekly_Sales"]].sum()
    st.bar_chart(monthly_data)  



    # Yearly view of sales
    st.header('*Yearly view of sales*')
    plt.figure(figsize=(10,6))
    newd = data.groupby("Year")[["Weekly_Sales"]].sum()
    # plt.xlabel("Years")
    # plt.ylabel("Weekly Sales")
    # plt.title("Yearly view of sales")
    st.bar_chart(newd)



    # find outliers could be another sidebar
    # fig, axs = plt.subplots(4,figsize=(5,12))
    # X = data[['Temperature','Fuel_Price','CPI','Unemployment']]
    # for i,column in enumerate(X):
    #     sns.boxplot(data[column], ax=axs[i])
    # st.pyplot(fig)



    # drop the outliers     
    data_new = data[(data['Unemployment']<10) & (data['Unemployment']>4.5) & (data['Temperature']>10)]
    # data_new



    # check outliers for testing maybe in sidebar
    # fig, axs = plt.subplots(4,figsize=(5,12))
    # X = data_new[['Temperature','Fuel_Price','CPI','Unemployment']]
    # for i,column in enumerate(X):
    #     sns.boxplot(data_new[column], ax=axs[i])
    # st.pyplot(fig)



    # Import sklearn 
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.linear_model import LinearRegression

    # Select features and target 
    X = data_new[['Store','Fuel_Price','CPI','Unemployment','Day','Month','Year']]
    y = data_new['Weekly_Sales']

    # Split data to train and test (0.80:0.20)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    tab1,tab2,tab3,tab4 = st.tabs(['Linear Regression','Random Forest Regressor','Knn Regressor','Gradient Boosting Regressor'])
    # Linear Regression model
    with tab1:
        st.header('*Linear Regression*:')
        st.write()
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        st.subheader('Metrics',divider='blue')
        st.write('*Accuracy*:',reg.score(X_train, y_train)*100)


        st.write('*Mean Absolute Error*:', metrics.mean_absolute_error(y_test, y_pred))
        st.write('*Mean Squared Error*:', metrics.mean_squared_error(y_test, y_pred))
        st.write('*Root Mean Squared Error*:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


        fig = plt.figure(figsize=(10,5))
        plt.scatter(y_test, y_test, c='b', label='Actual Values', marker='o', edgecolors='k')
        # Create a scatter plot for y_pred
        plt.scatter(y_test, y_pred, c='r', label='Predicted Values', marker='x')

        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        st.pyplot(fig)




    # Random Forest Regressor
    with tab2:
        st.header('*Random Forest Regressor*:')
        st.write()
        rfr = RandomForestRegressor(n_estimators = 400,max_depth=15,n_jobs=5)        
        rfr.fit(X_train,y_train)
        y_pred=rfr.predict(X_test)
        st.subheader('Metrics',divider='blue')
        st.write('*Accuracy*:',rfr.score(X_test, y_test)*100)

        st.write('*Mean Absolute Error*:', metrics.mean_absolute_error(y_test, y_pred))
        st.write('*Mean Squared Error*:', metrics.mean_squared_error(y_test, y_pred))
        st.write('*Root Mean Squared Error*:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        fig = plt.figure(figsize=(10,5))
        plt.scatter(y_test, y_test, c='b', label='Actual Values', marker='o', edgecolors='k')
        # Create a scatter plot for y_pred
        plt.scatter(y_test, y_pred, c='r', label='Predicted Values', marker='x')

        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        st.pyplot(fig)




    #Knn Regressor
    with tab3:
        st.header('*Knn Regressor*:')
        st.write()
        from sklearn.neighbors import KNeighborsRegressor
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        st.subheader('Metrics',divider='blue')
        st.write('*Accuracy*: {:2f}'.format(knn.score(X_test, y_test)*100))

        st.write('*Mean Absolute Error*: {:2f}'.format( metrics.mean_absolute_error(y_test, y_pred)))
        st.write('*Mean Squared Error*: {:2f}'.format( metrics.mean_squared_error(y_test, y_pred)))
        st.write('*Root Mean Squared Error*: {:2f}'.format( np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
        fig = plt.figure(figsize=(10,5))
        plt.scatter(y_test, y_test, c='b', label='Actual Values', marker='o', edgecolors='k')
        # Create a scatter plot for y_pred
        plt.scatter(y_test, y_pred, c='r', label='Predicted Values', marker='x')

        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        st.pyplot(fig)




    #Gradient Boosting Regressor
    with tab4:
        st.header('*Gradient Boosting Regressor*')
        st.write()
        from sklearn.ensemble import GradientBoostingRegressor
        gbr = GradientBoostingRegressor()
        gbr.fit(X_train,y_train)
        y_pred=gbr.predict(X_test)
        st.subheader('Metrics',divider='blue')
        # st.write('*Accuracy*:',gbr.score(X_test, y_test)*100)
        st.write('*Accuracy*: {:.2f}'.format(gbr.score(X_test, y_test)*100))

        st.write('*Mean Absolute Error*: {:2f}'.format( metrics.mean_absolute_error(y_test, y_pred)))
        st.write('*Mean Squared Error*: {:2f}'.format( metrics.mean_squared_error(y_test, y_pred)))
        st.write('*Root Mean Squared Error*: {:2f}'.format( np.sqrt(metrics.mean_squared_error(y_test, y_pred))))

        fig = plt.figure(figsize=(10,5))
        #I need different Colors for ytest and ypred
        # plt.scatter(x=y_test,y=y_pred,color=['red','blue'])
        plt.scatter(y_test, y_test, c='b', label='Actual Values', marker='o', edgecolors='k')
        # Create a scatter plot for y_pred
        plt.scatter(y_test, y_pred, c='r', label='Predicted Values', marker='x')

        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        st.pyplot(fig)