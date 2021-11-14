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

from app import app
from .reviewed_books import reviewed_books
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
            value="0",
            clearable=False
        )
    ),

    dcc.Graph(
        id='feedback_graph',
    ),

])
'''
Filters dewey list based on value of dewey filter
'''
@app.callback(
    Output("dewey_list_dropdown", "options"),
    Input("dewey_filter_dropdown", "value")
)
def update_dewey_list_options(selected_dewey_level):
    dewey_list = []
    selected_dewey_option = level_to_dewey_option(selected_dewey_level)

    if not selected_dewey_option == 'BC' and not selected_dewey_option == 'Nuevo':
        dewey_list = reviewed_books[selected_dewey_option].unique()
        dewey_list = [{"label": x, "value": x } for x in dewey_list]
    else:
        dewey_list=[]
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
    dewey_unit_name = level_to_dewey_option(dewey_unit)
    if not dewey_unit_name == 'BC' and not dewey_unit_name == 'Nuevo':
        selected_row: pd.DataFrame = reviewed_books.loc[(reviewed_books['Nivel'] == float(dewey_unit)) & (reviewed_books[dewey_unit_name]== dewey)]
    else:
        selected_row: pd.DataFrame  = reviewed_books.loc[(reviewed_books['Nivel'] == dewey_unit) ]
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

#Dado un valor numerico de dewey, se devuelve el nombre de ese valor
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

