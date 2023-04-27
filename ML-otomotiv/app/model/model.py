import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

def generate_month_list(start_date: int, end_date: int):
    # Convert the start and end dates to datetime objects
    start_date = datetime.strptime(str(start_date), '%Y%m%d')
    end_date = datetime.strptime(str(end_date), '%Y%m%d')

    # Initialize an empty list to store the months
    month_list = []

    # Loop through all the months between the start and end dates
    while start_date <= end_date:
        # Append the current month to the list
        month_list.append(start_date.strftime('%Y-%m-%d'))

        # Increment the month by 1
        if start_date.month == 12:
            start_date = start_date.replace(year=start_date.year+1, month=1)
        else:
            start_date = start_date.replace(month=start_date.month+1)

    # Return the list of months
    return month_list

with open(f"{BASE_DIR}/otv_model.pkl", 'rb') as f:
    otv_model = pickle.load(f)
with open(f"{BASE_DIR}/faiz_model.pkl", 'rb') as f:
    faiz_model = pickle.load(f)
with open(f"{BASE_DIR}/eur_model.pkl", 'rb') as f:
    eur_model = pickle.load(f)
with open(f"{BASE_DIR}/kredi_model.pkl", 'rb') as f:
    kredi_model = pickle.load(f)
with open(f"{BASE_DIR}/lr_model.pkl", 'rb') as f:
    lr_model = pickle.load(f)



def predict_pipeline(dates ):
    test_app= pd.DataFrame(generate_month_list(dates[0][0],dates[0][1]), columns=['Date'])
    test_app['Date'] = pd.to_datetime(test_app['Date'])
    new_cols = ['Otomotiv Satis','OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok']
    nan_values = np.empty((len(test_app), len(new_cols)))
    nan_values[:] = np.nan
    test_app[new_cols] = nan_values


    otv_future = otv_model.make_future_dataframe(periods=len(test_app), freq='M')
    otv_forecast = otv_model.predict(otv_future)

    faiz_future = faiz_model.make_future_dataframe(periods=len(test_app), freq='M')
    faiz_forecast = faiz_model.predict(faiz_future)

    eur_future = eur_model.make_future_dataframe(periods=len(test_app), freq='M')
    eur_forecast = eur_model.predict(eur_future)

    kredi_future = kredi_model.make_future_dataframe(periods=len(test_app), freq='M')
    kredi_forecast = kredi_model.predict(kredi_future)

   
    otv_pred = otv_forecast.iloc[-len(test_app):]['yhat'].values
    faiz_pred = faiz_forecast.iloc[-len(test_app):]['yhat'].values
    eur_pred = eur_forecast.iloc[-len(test_app):]['yhat'].values
    kredi_pred = kredi_forecast.iloc[-len(test_app):]['yhat'].values

    test_app = test_app[['OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok']]
    test_app['OTV Orani']=otv_pred.round(2)
    test_app['Faiz']=faiz_pred.round(2)
    test_app['EUR/TL']=eur_pred.round(2)
    test_app['Kredi Stok']=kredi_pred.round(2)


    y_pred = lr_model.predict(test_app).round(2)
        

    return list(y_pred)