import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def generate_LSTM_forecast(data, days, layers, neurons, learning_rate, epochs):
    scaler = MinMaxScaler()
    data.set_index('Date', inplace=True)
    scaled_prices = scaler.fit_transform(data)
    time_step = 60

    train_size = int(len(data) * 0.98)
    train_data = scaled_prices[0:train_size]

    X_train = []
    y_train = []
    for i in range(len(train_data) - time_step):
        X_train.append(train_data[i:i + time_step])
        y_train.append(train_data[i + time_step])
    X_train, y_train = np.array(X_train), np.array(y_train)

    test_data = scaled_prices[train_size - time_step:]
    X_test = []
    y_test = []
    for i in range(len(test_data) - time_step):
        X_test.append(test_data[i:i + time_step])
        y_test.append(test_data[i + time_step])
    X_test, y_test = np.array(X_test), np.array(y_test)

    model = Sequential()
    model.add(LSTM(neurons, return_sequences=(layers > 1), input_shape=(time_step, 1)))
    for i in range(1, layers):
        model.add(LSTM(neurons, return_sequences=(i < layers - 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, callbacks=[early_stopping])

    y_pred = model.predict(X_test)

    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    result_df = pd.DataFrame({
        'Date': data.iloc[-30:].index,
        'Actual': y_test.flatten()[-30:],
        'Predicted': y_pred.flatten()[-30:]
    })

    error = mean_squared_error(y_test, y_pred)

    future_date = pd.to_datetime(data.index[-1])
    last_sequence = scaled_prices[-time_step:].reshape(1, time_step, 1)
    future_forecasts = []
    future_dates = []

    for _ in range(days):
        forecast = model.predict(last_sequence)[0, 0]
        future_forecasts.append(forecast)
        last_sequence = np.append(last_sequence[:, 1:, :], [[[forecast]]], axis=1)
        future_date = (future_date + pd.Timedelta(days=1))
        future_dates.append(future_date)

    future_forecasts = np.array(future_forecasts).flatten()
    future_forecasts_reshaped = future_forecasts.reshape(-1, 1)
    future_forecasts = scaler.inverse_transform(future_forecasts_reshaped)
    future_forecasts = future_forecasts.flatten()
    forecast_df = pd.DataFrame({'Date': future_dates, 'Close': future_forecasts})
    return forecast_df, result_df, error