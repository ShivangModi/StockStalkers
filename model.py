import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error


# Simple Moving Average
def MovingAverage(df):
    ma_pred = []
    window_size = 100

    i = len(df) - window_size
    while i > 0:
        window = df['Close'][i: i + window_size]
        window_average = round(sum(window) / window_size, 2)
        ma_pred.append(window_average)
        i = i - 1
    ma_pred.reverse()
    return ma_pred


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
        return np.dot(x, self.m) + self.c


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
    params = {'n_neighbors': [i for i in range(2, 10)]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)

    # fit model and make prediction
    model.fit(x_train, y_train)

    x_scaled = scaler.fit_transform(x_test)
    x = pd.DataFrame(x_scaled)
    knn_pred = model.predict(x)
    return knn_pred


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
