import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM


# Simple Moving Average
def MovingAverage(df):
    ma_pred = []
    window_size = 200

    i = len(df) - window_size
    while i > 0:
        window = df['Close'][i: i + window_size]
        window_average = round(sum(window) / window_size, 2)
        ma_pred.append(window_average)
        i = i - 1
    ma_pred.reverse()

    pred = pd.DataFrame({'Date': df['Date'][200:], 'Value': df['Close'][200:], 'Predicted_Values': ma_pred})
    return pred


# Simple Linear Regression with Gradient Descent Algorithm
class LinearRegression:
    def __init__(self, df, learning_rate=0.0001, stop=1e-6, normalize=True):
        self.m = 0
        self.c = 0
        self.__df = df
        self.__lr = learning_rate
        self.__stop = stop
        self.__normalize = normalize
        self.__n = None
        self.__mean = None
        self.__std = None
        self.__costs = []
        self.__iterations = []

    def __computeCost(self, y_predict, y):
        loss = np.square(y_predict - y)
        cost = np.sum(loss) / (2 * self.__n)
        return cost

    def __optimize(self, x, y):
        y_predict = np.dot(x, self.m) + self.c
        dm = np.dot(x, (y_predict - y)) / self.__n
        dc = np.sum(y_predict - y) / self.__n
        self.m = self.m - self.__lr * dm
        self.c = self.c - self.__lr * dc

    def __normalizeX(self, x):
        return (x - self.__mean) / self.__std

    def fit(self):
        split = int(len(self.__df) * 0.75)
        train = self.__df[:split]

        x_train = np.array(train['Open'])
        y_train = np.array(train['Close'])

        if self.__normalize:
            self.__mean, self.__std = x_train.mean(axis=0), x_train.std(axis=0)
            x_train = self.__normalizeX(x_train)

            self.__n = len(x_train)
            last_cost, i = float('inf'), 0
            while True:
                y_predict = np.dot(x_train, self.m) + self.c
                cost = self.__computeCost(y_predict, y_train)
                self.__optimize(x_train, y_train)
                if last_cost - cost < self.__stop:
                    break
                else:
                    last_cost, i = cost, i + 1
                    self.__costs.append(cost)
                    self.__iterations.append(i)

    def predict(self, x):
        if self.__normalize:
            x = self.__normalizeX(x)
        slr_pred = np.dot(x, self.m) + self.c
        pred = pd.DataFrame({'Date': self.__df['Date'], 'Value': self.__df['Close'], 'Predicted_Values': slr_pred})
        return pred


# K-Nearest Neighbor algorithm with library
def KNN(df):
    split = int(len(df) * 0.75)
    train = df[:split]

    x_train = np.array(train['Date']).reshape(-1, 1)
    y_train = np.array(train['Close'])
    x_test = np.array(df['Date']).reshape(-1, 1)

    # scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(x_train)
    x_train = pd.DataFrame(x_train_scaled)

    # GrindSearch to find the best parameter
    params = {
        'n_neighbors': range(1, 100),
        'weights': ["uniform", "distance"]
    }
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params)

    # fit model and make prediction
    model.fit(x_train, y_train)

    x_scaled = scaler.fit_transform(x_test)
    x = pd.DataFrame(x_scaled)
    knn_pred = model.predict(x)
    pred = pd.DataFrame({'Date': df['Date'], 'Value': df['Close'], 'Predicted_Values': knn_pred})
    return pred


# Auto Regressive Integrated Moving Average (ARIMA)
def AR(p, df):
    # Generating the lagged p terms
    for i in range(1, p + 1):
        df['Shifted_values_%d' % i] = df['Value'].shift(i)

    # Breaking data set into test and training
    split = int(len(df) * 0.75)
    df_train = pd.DataFrame(df[0:split])
    df_test = pd.DataFrame(df[split:df.shape[0]])

    df_train_2 = df_train.dropna()
    # X contains the lagged values, hence we skip the first column
    X_train = df_train_2.iloc[:, 1:].values.reshape(-1, p)
    # Y contains the value, it is the first column
    y_train = df_train_2.iloc[:, 0].values.reshape(-1, 1)

    # Running linear regression to generate the co-efficients of lagged terms
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    theta = lr.coef_.T
    intercept = lr.intercept_
    df_train_2['Predicted_Values'] = X_train.dot(theta) + intercept

    X_test = df_test.iloc[:, 1:].values.reshape(-1, p)
    df_test['Predicted_Values'] = X_test.dot(theta) + intercept

    rmse = np.sqrt(mean_squared_error(df_test['Value'], df_test['Predicted_Values']))
    return [df_train_2, df_test, theta, intercept, rmse]


def MA(q, res):
    # Generating the lagged q terms
    for i in range(1, q + 1):
        res['Shifted_values_%d' % i] = res['Residuals'].shift(i)

    split = int(len(res) * 0.75)
    res_train = pd.DataFrame(res[0:split])
    res_test = pd.DataFrame(res[split:res.shape[0]])

    res_train_2 = res_train.dropna()
    X_train = res_train_2.iloc[:, 1:].values.reshape(-1, q)
    y_train = res_train_2.iloc[:, 0].values.reshape(-1, 1)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    theta = lr.coef_.T
    intercept = lr.intercept_
    res_train_2['Predicted_Values'] = X_train.dot(theta) + intercept

    X_test = res_test.iloc[:, 1:].values.reshape(-1, q)
    res_test['Predicted_Values'] = X_test.dot(theta) + intercept

    rmse = np.sqrt(mean_squared_error(res_test['Residuals'], res_test['Predicted_Values']))
    return [res_train_2, res_test, theta, intercept, rmse]


def ARIMA(data):
    df = pd.DataFrame(data['Close'])
    df.columns = ['Value']

    # find d value
    df_testing = pd.DataFrame(np.log(df.Value).diff().diff(12))

    # calculate the error on test for each p, and pick the best one
    best_p = -1
    best_rmse = 100000000000
    for i in range(1, 21):
        [df_train, df_test, theta, intercept, rmse] = AR(i, pd.DataFrame(df_testing.Value))
        if rmse < best_rmse:
            best_rmse = rmse
            best_p = i
    [df_train, df_test, theta, intercept, rmse] = AR(best_p, pd.DataFrame(df_testing.Value))
    df_c = pd.concat([df_train, df_test])

    res = pd.DataFrame()
    res['Residuals'] = df_c.Value - df_c.Predicted_Values

    # find q value
    best_q = -1
    best_rmse = 100000000000
    for i in range(1, 13):
        [res_train, res_test, theta, intercept, rmse] = MA(i, pd.DataFrame(res.Residuals))
        if rmse < best_rmse:
            best_rmse = rmse
            best_q = i
    [res_train, res_test, theta, intercept, rmse] = MA(best_q, pd.DataFrame(res.Residuals))
    res_c = pd.concat([res_train, res_test])

    df_c.Predicted_Values += res_c.Predicted_Values
    df_c.Value += np.log(df).shift(1).Value
    df_c.Value += np.log(df).diff().shift(12).Value
    df_c.Predicted_Values += np.log(df).shift(1).Value
    df_c.Predicted_Values += np.log(df).diff().shift(12).Value
    df_c.Value = np.exp(df_c.Value)
    df_c.Predicted_Values = np.exp(df_c.Predicted_Values)

    df_c = df_c.dropna().reset_index()
    df_c.index = df_c['Date']
    return df_c[['Date', 'Value', 'Predicted_Values']]


# Long Short-Term Memory (LSTM)
def create_dataset(data, time_step=1):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i: (i + time_step), 0]  # i=0
        dataX.append(a)  # 0,1,2,3,4---99
        dataY.append(data[i + time_step, 0])  # 100
    return np.array(dataX), np.array(dataY)


def LSTM_model(data):
    df = data['Close']

    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df).reshape(-1, 1))

    split = int(len(df1) * 0.75)
    train_data = df1[:split, :]
    test_data = df1[split:, :1]

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    # X_test, y_test = create_dataset(test_data, time_step)
    X_test, y_test = create_dataset(df1, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=64, verbose=1)

    lstm_pred = model.predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred)
    lstm_pred = np.array(lstm_pred).ravel()
    pred = pd.DataFrame({'Date': data['Date'][101:], 'Value': data['Close'][101:], 'Predicted_Values': lstm_pred})

    x_input = df1[len(df1) - 100:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    # prediction for next 180 days
    future_value = []
    n_steps = 100
    i = 0
    while i < 180:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            future_value.extend(yhat.tolist())
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            future_value.extend(yhat.tolist())
        i = i + 1

    from datetime import datetime
    future_date = pd.date_range(datetime.today(), periods=180).tolist()
    future_value = scaler.inverse_transform(future_value)
    future_value = np.array(future_value).ravel()
    future_pred = pd.DataFrame({'Date': future_date, 'Value': future_value})

    return [pred, future_pred]
