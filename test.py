import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import csv
import datetime
from math import sqrt
from sklearn.svm import SVR
import sklearn.svm as svm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

column_names = [
    'Fran Datum Tid (UTC)', 'till', 'jour', 'temperature', 'Kvalitet', 'Tidsutsnitt:', 'Unnamed: 5'
]
column_names_used = [
    'Fran Datum Tid (UTC)', 'till', 'jour'
]
def make_numeric_values(arr, title):
    new_arr = []
    for date in arr[title]:
        new_date = make_date(date)
        new_arr.append(new_date)
    arr[title] = new_arr

def fix_array(arr):
    for name in column_names_used:
        make_numeric_values(arr, name)

def make_date(date):
    new_date = date.split(' ')
    new_date = new_date[0]
    new_date = new_date.split('-')
    new_number = ''
    first = True
    for number in new_date:
        if first:
            first = False
        else:
            new_number = new_number + number
    return new_number

def convert_date_to_string(plus_days):
    date = datetime.datetime.today() + timedelta(days=plus_days)
    date = date.strftime("%Y-%m-%d %H:%M:%S") 
    date = date.split(' ')
    date = date[0]
    date = date.split('-')
    date = date[1]+date[2]
    return date


def train():
    dataset_url1 = 'https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/2/station/71420/period/corrected-archive/data.csv'
    dataset_url2 = 'https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/2/station/71420/period/latest-months/data.csv'

    data1 = pd.read_csv(dataset_url1, sep=';', skiprows=3607, names=column_names)
    data2 = pd.read_csv(dataset_url2, sep=';', skiprows=15, names=column_names)

    data1 = data2.append(data1)
    data1 = data1.drop('Tidsutsnitt:', axis=1)
    X = data1.drop(["temperature"], axis=1)
    X = X.drop(['Kvalitet'], axis = 1)
    X = X.drop(['Unnamed: 5'], axis = 1)
    fix_array(X)

    y = data1['temperature']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

    tree_model = DecisionTreeRegressor()

    tree_model.fit(X_train, y_train)
    joblib.dump(tree_model, 'weather_predictor.pkl')
    print("-" * 48)
    print("\nDone training\n")
    print("-" * 48)

def predict_weather():
    tree_model = joblib.load('weather_predictor.pkl')

    print("-" * 48)
    print("Entrez les détails de la date que vous souhaitez prévoir")
    print("\n")
    option = input("Année: ")
    year = option
    option = input("Numéro de mois (00) : ")
    month = option
    option = input("Numéro du jour (00) : ")
    theday = option

    day = str(month) + str(theday)
    
    date = [
        [day, 
        (str(int(day) + 1)), 
        (day)]
    ]
    temp = tree_model.predict(date)[0]
    print("-" * 48)
    print("\nLa température est estimée : " + str(temp) + "\n")
    print("-" * 48)

def run_program(option):
    if option == 1:
        print("1")
    elif option == 2:
        predict_weather()

def run_menu():
    print("*" *48)
    print("-" *10 + " Que souhaitez-vous faire ?  " + "-" * 10)
    print("\n")
    print("1. Consulter la météo d'un jour précis")
    print("2. Prévoir le temps qu'il fera un jour précis")
    print("\n")

    option = input("Saisissez l'option : ")

    while True:
        if option == 2 or option == 1 or option == 9 or option == 3:
            break
        option = input("Saisissez l'option : ")
    return option

if __name__== "__main__":
    
    while True:
        option = run_menu()
        if option == 9:
            break
        if option == 3:
            train()
        else:
            run_program(option)