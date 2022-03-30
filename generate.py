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
                ),
            ]
        ),
        style={
            "margin-top": "15px"
        }
    )
    return stock_graph
