import dash
from dash import Input, Output, State

from stock import *
from models import *
from functions import *
from components import *

# # Importing dataset
# data = yf.download('TTM').reset_index()
# # data = yf.download('GOOGL').reset_index()
# # data = yf.download('SPY').reset_index()
# data.index = data['Date']
# # print(data)
#
# # Splitting dataset into train and test set
# split = int(len(data) * 0.7)
# train = data[:split]
# test = data[split:]
#
# x_train = np.array(train['Open'])
# y_train = np.array(train['Close'])
# x_test = np.array(test['Open'])
# y_test = np.array(test['Close'])
#
# # Linear Regression
# linear_regression = LinearRegression()
# linear_regression.fit(x_train, y_train)
# y_pred = linear_regression.predict(data["Open"])  # Test Prediction
# actual = go.Scatter(
#     x=data['Date'],
#     y=data['Close'],
#     name="Actual Close Value"
# )
# lr_prediction = go.Scatter(
#     x=data['Date'],
#     y=y_pred,
#     name="Linear Regression Prediction"
# )
#
# dt = [actual, lr_prediction]
# dt = []
# layout = dict(title="Stock Prediction", showlegend=False, height=500, width=700)
# fig = dict(data=dt, layout=layout)

desc = """
Stock Stalkers is a stock market prediction website.
The stock market process is full of uncertainty, expectations and is affected by 
many factors. Hence Stock market prediction is one of the important factors in 
finance and business.
In this we will predict price using different machine learning model like 
Moving Average(MA), Simple Linear Regression(LR), K-Nearest Neighbors(KNN), 
Auto-Regressive Integrated Moving Average(ARIMA) and Long Short-Term Memory(LSTM).
"""

# Initialising dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SUPERHERO],
    title="Stock Stalkers"
)

app.layout = html.Div(
    [
        navbar,
        # html.Div(
        #     dcc.Graph(
        #         id="Stock Prediction",
        #         figure=fig
        #     )
        # ),
        html.Div(
            id="main-view",
            style={
                "padding-top": "15px",
                "padding-right": "15px"
            }
        ),
    ],
    style={
        "margin": "20px",
    }
)


# add callback for toggling the collapse on small screens
@app.callback(
    Output(component_id="navbar-collapse", component_property="is_open"),
    Input(component_id="navbar-toggler", component_property="n_clicks"),
    State(component_id="navbar-collapse", component_property="is_open"),
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# add callback for ticker
@app.callback(
    Output(component_id="main-view", component_property="children"),
    [
        Input(component_id="ticker", component_property="value"),
        # Input(component_id="forecast-btn", component_property="n_clicks"),
    ],
    # [
    #     State(component_id="ticker", component_property="value"),
    #     State(component_id="forecast-day", component_property="value"),
    #     State(component_id="stock-date-picker-range", component_property="start_date"),
    #     State(component_id="stock-date-picker-range", component_property="end_date"),
    # ]
)
def update_ticker(ticker):
    # fetching context to determine which button triggered callback
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == "ticker":
        if ticker:
            stock = Stock(ticker)
            stock_info = stock.get_info()
            stock_detail = generate_stock_detail(stock_info)

            close_graph = stock.get_graph()
            stock_graph = generate_stock_graph(close_graph)

            return dbc.Row([stock_detail, stock_graph], align="center")
            # return stock_detail
        else:
            alert = dbc.Alert(
                "No TICKER found!!!",
                id="alert-fade",
                color="primary",
                dismissable=True
            )
            return alert


if __name__ == '__main__':
    # # plot
    # plt.plot(train['Date'], train['Close'])
    # plt.plot(test['Date'], test['Close'])
    # plt.plot(test['Date'], y_pred)
    # plt.show()

    app.run_server(debug=True)
