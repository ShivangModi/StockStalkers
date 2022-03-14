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
                            # width="auto",
                            align="center"
                        ),
                        dbc.Col(
                            html.H4(info["name"]),
                            # width="auto",
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
                            html.H6(info["description"])
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


def generate_stock_graph(graph):
    stock_graph = dbc.Col(
        [
            dbc.Row(
                [
                    dbc.Label("Prediction Model"),
                    dbc.Checklist(
                        options=[
                            {"label": "MA", "value": "MA"},
                            {"label": "LR", "value": "LR"},
                            {"label": "KNN", "value": "KNN"},
                            {"label": "ARIMA", "value": "ARIMA"},
                            {"label": "LSTM", "value": "LSTM"},
                        ],
                        value=[],
                        inline=True,
                        switch=True,
                        id="switches-inline-input",
                    )
                ],
                style={
                    "margin-bottom": "15px"
                }
            ),
            dbc.Row(
                dcc.Graph(figure=graph),
                align="center",
            )
        ],
        style={
            "padding": "15px"
        }
    )
    return stock_graph
