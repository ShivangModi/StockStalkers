import dash
from dash import Input, Output, State

from components import *

from stock import Stock
from generate import generate_stock_detail, generate_stock_graph, generate_score_table

# Initialising dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SUPERHERO],
    title="Stock Stalkers"
)

app.layout = html.Div(
    [
        navbar,
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
        Input(component_id="forecast-btn", component_property="n_clicks"),
    ],
    [
        State(component_id="ticker", component_property="value"),
    ]
)
def update_ticker(submit, ticker):
    # fetching context to determine which button triggered callback
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if submit and trigger == "forecast-btn":
        if ticker:
            stock = Stock(ticker)
            stock_info = stock.get_info()

            stock_detail = generate_stock_detail(stock_info)    # Generate Stock Details

            close_graph = stock.get_graph()
            stock_graph = generate_stock_graph(close_graph)     # Generate Prediction Graph

            table = generate_score_table(stock.r2)              # Generate Table

            # return dbc.Row([stock_detail, stock_graph], align="center")
            return dbc.Container(
                [
                    dbc.Row(stock_detail, align="center"),
                    dbc.Row(stock_graph, align="center"),
                    dbc.Row(table, align="center"),
                ]
            )
        else:
            alert = dbc.Alert(
                "No TICKER found!!!",
                id="alert-fade",
                color="primary",
                dismissable=True
            )
            return alert


if __name__ == '__main__':
    app.run_server(debug=True)
