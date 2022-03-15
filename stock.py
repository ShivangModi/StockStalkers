import yfinance as yf
import plotly.express as px
from models import *


class Stock:
    def __init__(self, ticker):
        # start_date: str = str(date.today() - timedelta(days=90))
        # end_date: str = str(date.today())

        self.df = None
        self.ticker = ticker
        self.__ma = None
        self.__slr = None
        self.__knn = None
        self.__arima = None
        self._lstm = None

        # self.start_date = start_date
        # self.end_date = end_date

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
        except Exception as e:
            raise ConnectionError

    def __get_data(self):
        try:
            # df = yf.download(
            #     tickers=self.ticker
            #     # start = self.start_date,
            #     # end = self.end_date
            # ).reset_index()
            stock = yf.Ticker(self.ticker)
            df = stock.history(period="max").reset_index()
            df.index = df['Date']
            self.df = df
            return True
        except Exception as e:
            print(e)
            return

    # def __actual_graph(self):
    #     fig = px.line(
    #         self.df,
    #         x="Date",
    #         y="Close",
    #         title="Close Price",
    #         height=450,
    #         width=600
    #     )
    #     return fig

    def __prediction(self):
        split = int(len(self.df) * 0.8)
        train = self.df[:split]

        x_train = np.array(train['Open'])
        y_train = np.array(train['Close'])

        slr = LinearRegression()
        slr.fit(x_train, y_train)
        slr_pred = slr.predict(self.df['Open'])

        return {'Actual Close Price': self.df['Open'], 'LR Prediction': slr_pred}

    def get_graph(self):
        if self.__get_data():
            # actual_graph = self.__actual_graph()
            close = self.__prediction()

            data = []
            for i in close:
                temp = dict(x=self.df["Date"], y=close.get(i), type='scatter', name=i)
                data.append(temp)

            fig = {
                'data': data,
                'layout': {
                    'title': 'Stock prediction'
                }
            }
            return fig
        else:
            raise ConnectionError
