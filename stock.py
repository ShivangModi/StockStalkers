import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go


class Stock:
    def __init__(self, ticker):
        # start_date: str = str(date.today() - timedelta(days=90))
        # end_date: str = str(date.today())

        self.df = None
        self.ticker = ticker
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

    def __close_graph(self):
        fig = px.line(
            self.df,
            x="Date",
            y="Close",
            title="Close Price",
            height=450,
            width=600
        )
        # fig = go.Scatter(
        #     x=self.df["Date"],
        #     y=self.df["Close"],
        #     name="Close"
        # )
        return fig

    def get_graph(self):
        if self.__get_data():
            close_graph = self.__close_graph()
            return close_graph
        else:
            raise ConnectionError
