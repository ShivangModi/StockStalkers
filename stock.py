import yfinance as yf

from math import sqrt

from model import *
from sklearn.metrics import r2_score


class Stock:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = None

        self.__data = []
        self.__actual = None
        self.__ma = None
        self.__slr = None
        self.__knn = None
        self.__arima = None
        self.__lstm = None
        self.__future = None

    def get_info(self):
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            info = {
                "logo": info["logo_url"],
                "name": info["longName"],
                "description": info["longBusinessSummary"],
                "city": info["city"],
                "country": info["country"],
                "website": info["website"]
            }
            return info
        except Exception:
            raise ConnectionError

    def __get_data(self):
        try:
            df = yf.download(self.ticker).reset_index()
            df.index = df['Date']
            self.df = df.dropna()
            return True
        except Exception as e:
            print(e)
            return

    def __prediction(self):
        # Actual Price
        actual = dict(x=self.df['Date'], y=self.df['Close'], type='scatter',
                      name='Actual Price')

        # Simple Moving Average
        self.__ma = MovingAverage(self.df)
        ma_pred = dict(x=self.__ma['Date'], y=self.__ma['Predicted_Values'], type='scatter',
                       name='MA Prediction Price')

        # Simple Linear Regression with Gradient Descent Algorithm
        slr = LinearRegression(self.df)
        slr.fit()
        self.__slr = slr.predict(self.df['Open'])
        slr_pred = dict(x=self.__slr['Date'], y=self.__slr['Predicted_Values'], type='scatter',
                        name='SLR Prediction Price')
        slr_mse = mean_squared_error(self.__slr['Value'], self.__slr['Predicted_Values'])
        slr_r2 = r2_score(self.__slr['Value'], self.__slr['Predicted_Values'])

        # K-Nearest Neighbor algorithm with library
        self.__knn = KNN(self.df)
        knn_pred = dict(x=self.__knn['Date'], y=self.__knn['Predicted_Values'], type='scatter',
                        name='KNN Prediction Price')
        knn_mse = mean_squared_error(self.__knn['Value'], self.__knn['Predicted_Values'])
        knn_r2 = r2_score(self.__knn['Value'], self.__knn['Predicted_Values'])

        # Auto Regressive Integrated Moving Average (ARIMA)
        self.__arima = ARIMA(self.df)
        arima_pred = dict(x=self.__arima['Date'], y=self.__arima['Predicted_Values'], type='scatter',
                          name='ARIMA Prediction Price')
        arima_mse = mean_squared_error(self.__arima['Value'], self.__arima['Predicted_Values'])
        arima_r2 = r2_score(self.__arima['Value'], self.__arima['Predicted_Values'])

        # Long Short-Term Memory (LSTM) with library
        self.__lstm, self.__future = LSTM_model(self.df)
        lstm_pred = dict(x=self.__lstm['Date'], y=self.__lstm['Predicted_Values'], type='scatter',
                         name='LSTM Prediction Price')
        lstm_mse = mean_squared_error(self.__lstm['Value'], self.__lstm['Predicted_Values'])
        lstm_r2 = r2_score(self.__lstm['Value'], self.__lstm['Predicted_Values'])

        # 180 days Future Prediction
        future_pred = dict(x=self.__future['Date'], y=self.__future['Value'], type='scatter',
                           name='180 days Future Prediction Price')

        self.__data = [actual, slr_pred, knn_pred, ma_pred, arima_pred, lstm_pred, future_pred]

    def get_graph(self):
        if self.__get_data():
            self.__prediction()
            fig = {
                'data': self.__data,
                'layout': {
                    'title': 'Stock Prediction',
                    'xaxis': {
                        'title': 'Date'
                    },
                    'yaxis': {
                        'title': 'Close Price'
                    },
                }
            }
            return fig
        else:
            raise ConnectionError

    def get_score(self):
        # Simple Moving Average
        ma_mse = mean_squared_error(self.__ma['Value'], self.__ma['Predicted_Values'])
        ma_r2 = r2_score(self.__ma['Value'], self.__ma['Predicted_Values'])

        # Simple Linear Regression with Gradient Descent Algorithm
        slr_mse = mean_squared_error(self.__slr['Value'], self.__slr['Predicted_Values'])
        slr_r2 = r2_score(self.__slr['Value'], self.__slr['Predicted_Values'])

        # K-Nearest Neighbor algorithm with library
        knn_mse = mean_squared_error(self.__knn['Value'], self.__knn['Predicted_Values'])
        knn_r2 = r2_score(self.__knn['Value'], self.__knn['Predicted_Values'])

        # Auto Regressive Integrated Moving Average (ARIMA)
        arima_mse = mean_squared_error(self.__arima['Value'], self.__arima['Predicted_Values'])
        arima_r2 = r2_score(self.__arima['Value'], self.__arima['Predicted_Values'])

        # Long Short-Term Memory (LSTM) with library
        lstm_mse = mean_squared_error(self.__lstm['Value'], self.__lstm['Predicted_Values'])
        lstm_r2 = r2_score(self.__lstm['Value'], self.__lstm['Predicted_Values'])

        r2 = ["R2 Score", slr_r2, knn_r2, ma_r2, arima_r2, lstm_r2]
        mse = ["MSE", slr_mse, knn_mse, ma_mse, arima_mse, lstm_mse]
        rmse = ["RMSE", sqrt(slr_mse), sqrt(knn_mse), sqrt(ma_mse), sqrt(arima_mse), sqrt(lstm_mse)]
        return r2, mse, rmse
