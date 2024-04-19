import os
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from itertools import combinations
import statistics as st
import matplotlib.pyplot as plt
# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)
data = pd.read_csv('../Data/data.csv')
min_mape = []


# The goal of this project: How to apply scikit-learn library to fit linear models, use them for prediction, compare the models, and select the best one


#1. creating traning and test sets, then fitting model and predicting based on rating(one variable)
def prediction(i):
    X = np.array(data.rating**i).reshape(-1, 1)
    y = data.salary

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions_test = model.predict(X_test)

    mape_train = mape(y_test, predictions_test)
    min_mape.append(mape_train)
    

# Raising predictor value to the 'i' power and comparing results
prediction(1)
prediction(2)
prediction(3)
prediction(4)
print(min_mape)


#2.  creating traning and test sets, then fitting model and predicting based on many independent variables (the variables are assumed to be independent at this stage)
def m_prediction():
    X = data.drop('salary',axis=1)
    y = data.salary

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return print(*model.coef_, sep=', ')

#Checking m_prediction model coefficients
mpre = m_prediction()


# finding variables where the correlation coefficient is greater than 0.2. Then creating combination of options
cor = data.corr()
delete = cor[(cor.salary > 0.2) & (cor.salary < 1)].index.to_list()
to_drop = [*combinations(delete,1), *combinations(delete,2) ]


min2_mape = []

#3. To check how removal of variables with high correlation influences model quality
def c_prediction():

    X = data.drop('salary',axis=1)
    y = data.salary
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    model = LinearRegression()

    #Removing selected variables in each combination to check results
    for d in to_drop:
        model.fit(X_train.drop(columns = list(d)), y_train)
        predictions_test = model.predict(X_test.drop(columns = list(d)))
        mape_train2 = mape(y_test, predictions_test)
        min2_mape.append(mape_train2)

    return min2_mape

#to compare results
cpre = c_prediction()
#print(cpre)

#Model without high correlation variables and without negative salaries predictions (salary can not be negative)
def n_prediction(choose):
    X = data.drop(columns =['salary','age', 'experience'])
    y = data.salary
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions_test = np.array(model.predict(X_test))
    mymedian = st.median(y_train)
    if choose == 0:
        predictions_test[predictions_test < 0] = 0
    else:
        predictions_test[predictions_test < 0] = mymedian

    return mape(y_test, predictions_test)

# comparing two results: 1. Replacing negative values with '0'  2. Replacing negative values with median
mylist = []
mylist.append(n_prediction(0))
mylist.append(n_prediction(1))
print(round(min(mylist),5))

