import dash_bootstrap_components as dbc

from dash import html

symbol = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

# logo
logo = dbc.Row(
    [
        dbc.Col(html.Img(src=symbol, height="50px")),
        dbc.Col(dbc.NavbarBrand("Stock Stalkers", className="ms-3", style={"font-size": "30px"}))
    ],
    align="center",
    className="g-0"
)

# stock_form
stock_form = dbc.Row(
    [
        dbc.Col(
            dbc.Input(
                id="ticker",
                placeholder="TICKER",
                size="lg",
            ),
            width="auto",
            align="center"
        ),
        dbc.Col(
            dbc.Button(
                "Forecast",
                id="forecast-btn",
                color="primary",
                className="ms-3",
                size="lg",
            ),
            width="auto",
            align="center"
        )
    ],
    align="center",
    className="g-0 ms-auto mt-3 mt-md-2"
)

# navbar
navbar = dbc.Navbar(
    [
        html.A(
            logo,
            href="/",
            style={
                "textDecoration": "none",
                "margin-left": "1cm",
            }
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(
            stock_form,
            id="navbar-collapse",
            navbar=True,
            style={
                "margin-left": "1cm",
                "margin-right": "1cm",
            }
        )
    ],
    dark=True,
    color="dark",
    style={
        "padding-top": "25px",
        "padding-bottom": "25px",
        "margin-bottom": "15px",
    }
)
