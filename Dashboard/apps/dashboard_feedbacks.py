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

import dash_bootstrap_components as dbc
from app import app
import requests
"""
Importacion de datos
"""

#Dropdown para seleccion de dewey
dewey_filters = [
        {"label":"Dewey Unidad", "value": "0.5"},
        {"label":"Dewey Decena", "value": "0.2"},
        {"label":"Dewey Centena","value": "0.1"},
        {"label":"Baja Circulación", "value": "BC"},
        {"label":"Libros Nuevos", "value": "Nuevo"}
        ]
"""
HTML
"""
layout = html.Div(children=[
    html.H1(children='UAQUE: Feedback de los usuarios'),

    dbc.Spinner(children=[
        html.Div(
            dcc.Dropdown(
                id="dewey_filter_dropdown",
                value="BC",
                options=dewey_filters,
                clearable=False,
                searchable=False,
            )
        ),

        html.Div(
            dcc.Dropdown(
                id="dewey_list_dropdown",
                value='0',
                clearable=False
            )
        ),

        dcc.Graph(
            id='feedback_graph',
        ),

        ]),

])
'''
Filters dewey list based on value of dewey filter
'''
@app.callback(
    Output("dewey_list_dropdown", "options"),
    Input("dewey_filter_dropdown", "value")
)
def update_dewey_list_options(selected_dewey_level):
    smartuj_endpoint: str = 'localhost:8000/api'
    uso_biblioteca: str = 'suj-e-004'
    dashboardFeedback: str = 'DashboardFeedbackUtilsDeweyList'

    #Agrupamiento crear perfiles grupales  http://{{smartuj-endpoint}}/{{perfil-grupal}}/model
    url_dewey_list: str= 'http://'+smartuj_endpoint+'/'+uso_biblioteca+'/'+dashboardFeedback

    dewey_list = (requests.get(url=url_dewey_list, params={'selected_dewey_level': selected_dewey_level}))
    dewey_list = dewey_list.json()

    return dewey_list

'''
Update graph based on
dewey list value
'''
@app.callback(
    Output("feedback_graph", "figure"),
    [Input("dewey_list_dropdown", "value")],
    [Input("dewey_filter_dropdown", "value")]
)

def update_graph(dewey, dewey_unit):

    smartuj_endpoint: str = 'localhost:8000/api'
    uso_biblioteca: str = 'suj-e-004'
    dashboardFeedback: str = 'DashboardFeedback'

    #Agrupamiento crear perfiles grupales  http://{{smartuj-endpoint}}/{{perfil-grupal}}/model
    url_feedbacks: str= 'http://'+smartuj_endpoint+'/'+uso_biblioteca+'/'+dashboardFeedback

    selected_row= pd.DataFrame(requests.get(url=url_feedbacks, params={'dewey': dewey, 'dewey_unit': dewey_unit}).json())
    scores = selected_row.groupby(['Calificacion']).size().reset_index(name='count')
    if scores.empty:
        fig =  {
            "layout": {
                "xaxis": {
                    "visible": False
                },
                "yaxis": {
                    "visible": False
                },
                "annotations": [{
                    "text": "No matching data found",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": { "size": 28 }
                }]
            }
        }
    else:
        fig = px.pie(scores, values="count", names=['Dislike',  'No response','Like'])
    return fig

