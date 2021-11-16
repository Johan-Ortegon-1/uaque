from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import dash_bootstrap_components as dbc
from app import app

from apps import (
    dashboard_feedbacks,
    dashboard_grupos,
    dashboard_pertenencia,
    dashboard_feedback_por_dewey,
    dashboard_feedbacks_individual,
    # dashboard_generar_recomendaciones,
)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}
sidebar = html.Div(
    [
        html.H2("UAQUE", className="display-4"),
        html.Hr(),
        html.P("Dashboards disponibles", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink(
                    "Feedbacks grupales", href="/apps/feedbacks", active="exact"
                ),
                dbc.NavLink(
                    "Información de grupos", href="/apps/grupos", active="exact"
                ),
                dbc.NavLink(
                    "Pertenencias de usuarios", href="/apps/pertenencia", active="exact"
                ),
                dbc.NavLink(
                    "Generar recomendaciones",
                    href="/apps/generar_recomendaciones",
                    active="exact",
                ),
                dbc.NavLink("Feedbacks individuales", href="/apps/feedbackIndividual", active="exact"),
                dbc.NavLink("Feedbacks individuales por Dewey", href="/apps/feedbackDewey", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)
content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=True),
        sidebar,
        # content will be rendered in this element
        content,
    ]
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/apps/feedbacks":
        return dashboard_feedbacks.layout
    elif pathname == "/apps/grupos":
        return dashboard_grupos.layout
    elif pathname == "/apps/pertenencia":
        return dashboard_pertenencia.layout
    elif pathname == '/apps/feedbackDewey':
        return dashboard_feedback_por_dewey.layout
    elif pathname == '/apps/feedbackIndividual':
        return dashboard_feedbacks_individual.layout
    elif pathname == "/apps/generar_recomendaciones":
        pass  # return dashboard_generar_recomendaciones.layout
    else:
        return dbc.Jumbotron(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        )


if __name__ == "__main__":
    app.run_server(debug=False)
