from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from app import app
from apps import (
        dashboard_feedback_por_dewey,
        dashboard_feedbacks,
        dashboard_feedbacks_individual,
        dashboard_grupos,
        dashboard_pertenencia,
        home,
        )

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    # content will be rendered in this element
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/apps/feedbackDewey':
        return dashboard_feedback_por_dewey.layout
    elif pathname == '/apps/feedbacks':
        return dashboard_feedbacks.layout
    elif pathname == '/apps/feedbackIndividual':
        return dashboard_feedbacks_individual.layout
    elif pathname == '/apps/grupos':
        return dashboard_grupos.layout
    elif pathname == '/apps/pertenencia':
        return dashboard_pertenencia.layout
    elif pathname == '/home':
        return home.layout
    else:
        return ('404 ', pathname)

if __name__ == '__main__':
    app.run_server(debug=True)
