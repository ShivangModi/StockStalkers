import yfinance as yf

from model import *


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
        actual = dict(x=self.df['Date'], y=self.df['Close'], type='scatter', name='Actual Price')

        # Simple Moving Average
        self.__ma = MovingAverage(self.df)
        ma_pred = dict(x=self.df['Date'][100:], y=self.__ma, type='scatter', name='MA Prediction Price')

        # Simple Linear Regression with Gradient Descent Algorithm
        slr = LinearRegression(self.df)
        slr.fit()
        self.__slr = slr.predict(self.df['Open'])
        slr_pred = dict(x=self.df['Date'], y=self.__slr, type='scatter', name='SLR Prediction Price')

        # K-Nearest Neighbor algorithm with library
        self.__knn = KNN(self.df)
        knn_pred = dict(x=self.df['Date'], y=self.__knn, type='scatter', name='KNN Prediction Price')

        # Auto Regressive Integrated Moving Average (ARIMA)
        self.__arima = ARIMA(self.df)
        arima_pred = dict(x=self.__arima['Date'], y=self.__arima['Predicted_Values'], type='scatter',
                          name='ARIMA Prediction Price')

        # self.__lstm = dict(x=self.df['Date'], y=self.df['Close'], type='scatter', name='LSTM Price')

        self.__data = [actual, slr_pred, knn_pred, ma_pred, arima_pred]

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
