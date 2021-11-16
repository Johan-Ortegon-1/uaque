# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
'''
Dashboard that shows user groups with percentages
and recommended books
'''
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from app import app
import requests

"""
HTML
"""
layout = html.Div(children=[
    html.H1(children='UAQUE: Feedback de los usuarios individuales en cada Dewey'),

    html.Div(
        dcc.Dropdown(
            id="users_id_dropdown",
            clearable=False,
            searchable=True,
            value="6466dbb15c41fdacb59eb1179817958de2c57191",
        )
    ),

    dcc.Graph(
        id='feedback_dewey_graph',
    ),

])

#Cuando cambia el valor de busqueda, cambian las opciones que preesnta el dropdown.
@app.callback(
    Output("users_id_dropdown", "options"),
    Input("users_id_dropdown", "search_value")
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate

    smartuj_endpoint: str = 'localhost:8000/api'
    uso_biblioteca: str = 'suj-e-004'
    dashboardFpDOptions: str = 'DashboardFeedbackPorDeweyUtilsOption'

    #Agrupamiento crear perfiles grupales  http://{{smartuj-endpoint}}/{{perfil-grupal}}/model
    url_feedbacks_por_dewey: str= 'http://'+smartuj_endpoint+'/'+uso_biblioteca+'/'+dashboardFpDOptions

    id_users = (requests.get(url=url_feedbacks_por_dewey, params={'search_value':search_value}).json())
    #carga solo 50 resultados (no puede cargar los 40,000)
    return id_users

'''
Update graph based on
user list value
'''
@app.callback(
    Output("feedback_dewey_graph", "figure"),
    [Input("users_id_dropdown", "value")],
)

def update_graph(id_user):
    smartuj_endpoint: str = 'localhost:8000/api'
    uso_biblioteca: str = 'suj-e-004'
    dashboardFpD: str = 'DashboardFeedbackPorDewey'

    #Agrupamiento crear perfiles grupales  http://{{smartuj-endpoint}}/{{perfil-grupal}}/model
    url_feedbacks: str= 'http://'+smartuj_endpoint+'/'+uso_biblioteca+'/'+dashboardFpD

    selected_row= pd.DataFrame(requests.get(url=url_feedbacks, params={'id_user': id_user}).json())
    scores = selected_row.groupby(['Calificacion', 'DeweyUnidad']).size().reset_index(name='count')

    scores = scores[['DeweyUnidad', 'Calificacion', 'count']]
    fig = px.bar(scores, x="DeweyUnidad", y="count", color="Calificacion", color_discrete_map={
        '-1': 'red',
        '1': 'green',
        '0': 'blue',
    })
    fig.update_xaxes(type='category')
    return fig

