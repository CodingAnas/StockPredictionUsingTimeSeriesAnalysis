from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
from numpy import array

import matplotlib.pyplot as plt
import pandas as pd #for saving data to csv
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

load_dotenv()

app = Flask(__name__)
CORS(app)

def create_dataset(dataset, timesteps):
    dataX = []
    dataY = []
    for i in range(0, len(dataset) - timesteps - 1):
        a = dataset[i:(i+timesteps), 0]
        dataX.append(a)
        dataY.append(dataset[i+timesteps, 0])
    return np.array(dataX), np.array(dataY)

@app.route("/predict")
def predict():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker not provided"}), 400
    api_key = os.getenv('TIINGO_API_KEY')
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {api_key}'
    }

    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"

    params = {
        'startDate': start_date,
        'endDate': end_date,
        'format': 'csv',
        'resampleFreq': 'daily'
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        with open(f"{ticker}_tiingo_data_{start_date}_to_{end_date}.csv", "wb") as file:
            file.write(response.content)
        print(f"Data saved to {ticker}_tiingo_data_{start_date}_to_{end_date}.csv")
    else:
        print(f"Error {response.status_code}: {response.text}")

    file = f"{ticker}_tiingo_data_{start_date}_to_{end_date}.csv"

    df = pd.read_csv(file)

    if response.status_code == 200:
        with open(f"{ticker}_tiingo_data_{start_date}_to_{end_date}.csv", "wb") as file:
            file.write(response.content)
        print(f"Data saved to {ticker}_tiingo_data_{start_date}_to_{end_date}.csv")
    else:
        print(f"Error {response.status_code}: {response.text}")

    file = f"{ticker}_tiingo_data_{start_date}_to_{end_date}.csv"

    df = pd.read_csv(file)

    df1 = df.reset_index()["close"]

    df2 = np.array(df1).reshape(-1, 1)

    train_siz = int(len(df2)*0.65)
    test_siz = len(df2) - train_siz

    Train_DS = df2[:train_siz]
    Test_DS = df2[train_siz:]

    scaler = MinMaxScaler(feature_range=(0, 1))

    Train_DS_scaled = scaler.fit_transform(Train_DS)

    Test_DS_scaled = scaler.transform(Test_DS)

    timeSteps = 100
    x_train, y_train = create_dataset(Train_DS_scaled, timeSteps)
    X_test, Y_test = create_dataset(Test_DS_scaled, timeSteps)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(loss="mean_squared_error",optimizer='adam')

    model.summary()

    model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, Y_test), verbose=1)


    x_input=Test_DS_scaled[340:].reshape(1,-1)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=100
    i=0
    while(i<30):

        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    
    future_preds = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

    future_preds = future_preds.flatten().tolist()

    return jsonify({"ticker": ticker, "forecast": future_preds})


if __name__ == "__main__":
    app.run(port=5001, debug=True)





# y_pred = model.predict(X_test)

# y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
# y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))

# from sklearn.metrics import mean_squared_error
# import numpy as np

# rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
# print("Test RMSE:", rmse)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.plot(y_test_inv, label='Actual')
# plt.plot(y_pred_inv, label='Predicted')
# plt.title('LSTM Predictions vs Actual')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()

# y_train_pred = model.predict(x_train)

# y_train_pred_inv = scaler.inverse_transform(y_train_pred.reshape(-1, 1))
# y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))

# from sklearn.metrics import mean_squared_error
# import numpy as np

# rmse = np.sqrt(mean_squared_error(y_train_inv, y_train_pred_inv))
# print("Test RMSE:", rmse)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.plot(y_train_inv, label='Actual')
# plt.plot(y_train_pred_inv, label='Predicted')
# plt.title('LSTM trained data Predictions vs Actual')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.show()

# Test_DS_scaled.shape

# len(Test_DS_scaled)

# x


# print(lst_output)

# lst_output

# day_new=np.arange(1,101)
# day_pred=np.arange(101,131)

# len(df2)

# df2



# actual_last_100 = df2[-100:]

# actual_last_100.shape, future_preds.shape

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10,6))
# plt.plot(day_new, actual_last_100, label='Last 100 Actual Data')
# plt.plot(day_pred, future_preds, label='30-Day Forecast')
# plt.xlabel('Days')
# plt.ylabel('Price')
# plt.title('Actual Data vs Future Forecast')
# plt.legend()
# plt.grid(True)
# plt.show()


