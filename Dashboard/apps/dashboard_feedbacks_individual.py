# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
'''
Dashboard that shows user groups with percentages
and recommended books
'''
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from app import app
import requests
#from .reviewed_books import reviewed_books
"""
Importacion de datos
"""
#id_users = [{"label": x, "value": x } for x in reviewed_books["IDUsuario"].unique()]

"""
HTML
"""
layout = html.Div(children=[
    html.H1(children='UAQUE: Feedback de los usuarios individuales'),

    html.Div(
        dcc.Dropdown(
            id="individual_users_id_dropdown",
            clearable=False,
            searchable=True,
            value="6466dbb15c41fdacb59eb1179817958de2c57191",
        )
    ),

    dcc.Graph(
        id='feedback_user_graph',
    ),

])

#Cuando cambia el valor de busqueda, cambian las opciones que preesnta el dropdown.
@app.callback(
    Output("individual_users_id_dropdown", "options"),
    Input("individual_users_id_dropdown", "search_value")
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate

    smartuj_endpoint: str = 'localhost:8000/api'
    uso_biblioteca: str = 'suj-e-004'
    dashboardFIndividualOptions: str = 'DashboardFeedbackIndividualUtilsOption'


    #Agrupamiento crear perfiles grupales  http://{{smartuj-endpoint}}/{{perfil-grupal}}/model
    url_feedbacks_por_dewey: str= 'http://'+smartuj_endpoint+'/'+uso_biblioteca+'/'+dashboardFIndividualOptions

    id_users = (requests.get(url=url_feedbacks_por_dewey, params={'search_value':search_value}).json())
    #carga solo 50 resultados (no puede cargar los 40,000)
    return id_users

'''
Update graph based on
user list value
'''
@app.callback(
    Output("feedback_user_graph", "figure"),
    [Input("individual_users_id_dropdown", "value")],
)

def update_graph(id_user):

    smartuj_endpoint: str = 'localhost:8000/api'
    uso_biblioteca: str = 'suj-e-004'
    dashboardFIndividual: str = 'DashboardFeedbackIndividual'

    #Agrupamiento crear perfiles grupales  http://{{smartuj-endpoint}}/{{perfil-grupal}}/model
    url_feedbacks: str= 'http://'+smartuj_endpoint+'/'+uso_biblioteca+'/'+dashboardFIndividual

    selected_row= pd.DataFrame(requests.get(url=url_feedbacks, params={'id_user': id_user}).json())

    scores = selected_row.groupby(['Calificacion']).size().reset_index(name='count')
    fig = px.pie(scores, values="count", names=['Dislike',  'No response','Like'])
    return fig

