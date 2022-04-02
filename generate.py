import dash_bootstrap_components as dbc

from dash import html, dcc


def generate_stock_detail(info: dict):
    stock_details = dbc.Col(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(src=info["logo"]),
                            align="center"
                        ),
                        dbc.Col(
                            html.H4(info["name"]),
                            align="center"
                        )
                    ],
                    align="center",
                    style={
                        "margin-bottom": "10px",
                    }
                ),
                dbc.Row(
                    dbc.Col(
                        html.P(
                            html.H6(info["description"]),
                            style={
                                'text-align': 'justify',
                            }
                        )
                    )
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.P(
                                info["city"],
                                className="text-muted"
                            )
                        ),
                        dbc.Col(
                            html.P(
                                info["country"],
                                className="text-muted"
                            )
                        ),
                        dbc.Col(
                            html.A(
                                info["website"],
                                href=info["website"],
                                className="text-muted",
                                style={
                                    "textDecoration": "none",
                                }
                            ),
                        ),
                    ]
                )
            ]
        ),
        align="center"
    )
    return stock_details


def generate_stock_graph(fig):
    stock_graph = dbc.Col(
        dbc.Container(
            [
                dbc.Row(
                    dcc.Graph(figure=fig),
                    align="center",
                )
            ]
        ),
        style={
            "margin-top": "15px"
        }
    )
    return stock_graph


def generate_score_table(r2, mse, rmse):
    table_header = [
        html.Thead(
            html.Tr(
                [
                    html.Th(""),
                    html.Th("SLR"),
                    html.Th("KNN"),
                    html.Th("MA"),
                    html.Th("ARIMA"),
                    html.Th("LSTM"),
                ]
            )
        )
    ]

    # R2 Score
    row1 = html.Tr([html.Td(str(i)) for i in r2])
    row2 = html.Tr([html.Td(str(i)) for i in mse])
    row3 = html.Tr([html.Td(str(i)) for i in rmse])

    table_body = [html.Tbody([row1, row2, row3])]

    table = dbc.Col(
        dbc.Container(
            [
                dbc.Table(table_header + table_body, bordered=True)
            ]
        ),
        style={
            "margin-top": "15px"
        }
    )
    return table
