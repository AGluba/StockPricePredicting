import yfinance as yf
import pandas as pd
import datetime
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


def generate_data(market):
    file_path = f'data/{market}_{datetime.date.today()}.csv'

    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
    else:
        end_date = datetime.date.today()
        data = yf.download(f'{market}', start='2012-01-01', end=end_date, auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        data.to_csv(file_path, index=False)

    return data[['Date', 'Close']]


def generate_arima_forecast(data, days, p, d, q):
    future_date = data['Date'].iloc[-1]
    future_forecasts = []
    future_dates = []

    for _ in range(days):
        model = ARIMA(data['Close'], order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1).iloc[0]
        
        future_date = (future_date + pd.Timedelta(days=1))
        future_forecasts.append(forecast)
        future_dates.append(future_date)
        forecast_df = pd.DataFrame({'Close': [forecast]}, index=[future_date])
        data = pd.concat([data, forecast_df], axis=0, ignore_index=False)

    forecast_df = pd.DataFrame({'Date': future_dates, 'Close': future_forecasts})
    return forecast_df

def generate_report(data, p, d, q, days):
    train = data.iloc[:-30-days]
    test = data.iloc[-30-days:-days]

    predictions = []
    value_test = len(test)

    for i in range(value_test):
        model = ARIMA(train["Close"], order=(p, d, q))
        model_fit = model.fit()

        output = model_fit.forecast(steps=1)
        forecast = output.iloc[0]
        predictions.append(forecast)

        forecast_df = pd.DataFrame({'Close': test.iloc[i]['Close']}, index=[test.index[i]])
        train = pd.concat([train, forecast_df], axis=0)

    predictions_df = pd.DataFrame({'Close': predictions}, index=test.index)
    result_df = pd.DataFrame({
        'Date': test['Date'],
        'Actual': test['Close'].values,
        'Predicted': predictions_df['Close'].values
    })

    error = mean_squared_error(test['Close'], predictions_df['Close'])

    return result_df, error

