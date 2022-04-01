import yfinance as yf

from model import *
from sklearn.metrics import r2_score


class Stock:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = None
        self.r2 = []

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
            self.df = df
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
        ma_r2 = r2_score(self.__ma['Value'], self.__ma['Predicted_Values'])

        # Simple Linear Regression with Gradient Descent Algorithm
        slr = LinearRegression(self.df)
        slr.fit()
        self.__slr = slr.predict(self.df['Open'])
        slr_pred = dict(x=self.__slr['Date'], y=self.__slr['Predicted_Values'], type='scatter',
                        name='SLR Prediction Price')
        slr_r2 = r2_score(self.__slr['Value'], self.__slr['Predicted_Values'])

        # K-Nearest Neighbor algorithm with library
        self.__knn = KNN(self.df)
        knn_pred = dict(x=self.__knn['Date'], y=self.__knn['Predicted_Values'], type='scatter',
                        name='KNN Prediction Price')
        knn_r2 = r2_score(self.__knn['Value'], self.__knn['Predicted_Values'])

        # Auto Regressive Integrated Moving Average (ARIMA)
        self.__arima = ARIMA(self.df)
        arima_pred = dict(x=self.__arima['Date'], y=self.__arima['Predicted_Values'], type='scatter',
                          name='ARIMA Prediction Price')
        arima_r2 = r2_score(self.__arima['Value'], self.__arima['Predicted_Values'])

        # Long Short-Term Memory (LSTM) with library
        self.__lstm, self.__future = LSTM_model(self.df)
        lstm_pred = dict(x=self.__lstm['Date'], y=self.__lstm['Predicted_Values'], type='scatter',
                         name='LSTM Prediction Price')
        lstm_r2 = r2_score(self.__lstm['Value'], self.__lstm['Predicted_Values'])

        self.__data = [actual, slr_pred, knn_pred, ma_pred, arima_pred, lstm_pred]
        self.r2 = [slr_r2, knn_r2, ma_r2, arima_r2, lstm_r2]

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
