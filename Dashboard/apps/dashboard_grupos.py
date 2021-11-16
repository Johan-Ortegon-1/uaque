# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
'''
Dashboard that shows users in a particular
Dewey group
'''
from dash import dcc, html, dash_table
import pandas as pd
from dash.dependencies import Input, Output
import requests
from app import app
"""
Importacion de datos
"""

table_columns = ['nombre_usuario', 'email','IDUsuario', 'Facultad', 'Programa']


dewey_filters = [
        {"label":"Dewey Unidad", "value": "0.5"},
        {"label":"Dewey Decena", "value": "0.2"},
        {"label":"Dewey Centena","value": "0.1"},
        ]


layout = html.Div(children=[
    html.H1(children='Grupos UAQUE'),
    html.Div([
        dcc.Dropdown(
            id="dewey_filter_dropdown",
            value="0.5",
            options=dewey_filters,
            clearable=False,
            searchable=False,
        ),
        dcc.Dropdown(
            id="dewey_groups_list_dropdown",
            value="725",
            clearable=False
        ),
    ]),
    html.Br(),
    html.Div(
        id='count',
        ),
    html.Br(),
    dash_table.DataTable(
        id='users_table',
        columns=[{"name": i, "id": i} for i in table_columns],
        export_format='csv',
    ),
    html.Div(
        children=[
            html.Ul(id='', children=[html.Li(i) for i in []]),
        ],
    ),

])

'''
Filters dewey list based on value of dewey filter
'''
@app.callback(
    Output("dewey_groups_list_dropdown", "options"),
    Input("dewey_filter_dropdown", "value")
)
def update_dewey_list_options(selected_dewey_level):
    smartuj_endpoint: str = 'localhost:8000/api'
    uso_biblioteca: str = 'suj-e-004'
    dashboardFeedback: str = 'DashboardGruposUtilsDeweyList'

    #Agrupamiento crear perfiles grupales  http://{{smartuj-endpoint}}/{{perfil-grupal}}/model
    url_dewey_list: str= 'http://'+smartuj_endpoint+'/'+uso_biblioteca+'/'+dashboardFeedback

    dewey_list = (requests.get(url=url_dewey_list, params={'selected_dewey_level': selected_dewey_level}))
    dewey_list = dewey_list.json()

    return dewey_list


@app.callback(
    Output("users_table", "data"),
    [Input("dewey_groups_list_dropdown", "value")],
)

def update_table(dewey):

    smartuj_endpoint: str = 'localhost:8000/api'
    uso_biblioteca: str = 'suj-e-004'
    dashboardGrupos: str = 'Dashboard'

    url_grupos: str= 'http://'+smartuj_endpoint+'/'+uso_biblioteca+'/'+ dashboardGrupos

    selected_rows = requests.get(url=url_grupos, params={'dewey': dewey})
    selected_rows = selected_rows.json()
    return selected_rows

@app.callback(
        Output('count','children'),
        Input('users_table', 'data')
)

def update_count(user_table_data):
    return 'Total: {}'.format(len(user_table_data))

def level_to_dewey_option(selected_dewey_level):
    if selected_dewey_level == "0.5":
        selected_dewey_option = 'DeweyUnidad'
    elif selected_dewey_level == "0.2":
        selected_dewey_option = 'DeweyDecena'
    elif selected_dewey_level == "0.1":
        selected_dewey_option = 'DeweyCentena'
    elif selected_dewey_level == "BC":
        selected_dewey_option = 'BC'
    elif selected_dewey_level == "Nuevo":
        selected_dewey_option = 'BC'
    else:
        print("ERROR", selected_dewey_level)
        selected_dewey_option = 'DeweyUnidad'
    return selected_dewey_option


