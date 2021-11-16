import time
import requests
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd

from app import app

import dash_bootstrap_components as dbc

groupId :str= '6466dbb15c41fdacb59eb1179817958de2c57191'
smartuj_endpoint: str = 'localhost:8000/api'
recomendacion_uaque: str = 'suj-i-009'
perfil_grupal: str = 'suj-s-009'

#Agrupamiento crear perfiles grupales  http://{{smartuj-endpoint}}/{{perfil-grupal}}/model
url_agrupamiento: str= 'http://'+smartuj_endpoint+'/'+perfil_grupal+'/model'
#Entrenar modelo generar recomendaciones http://{{smartuj-endpoint}}/{{recomendacion-uaque}}/model
url_entrenamient: str= 'http://'+smartuj_endpoint+'/'+recomendacion_uaque+'/model'


table_header = [
    html.Thead(html.Tr([
        html.Th("itemId"),
        html.Th("title"),
        html.Th("location"),
        html.Th("author"),
        html.Th("userId"),
        html.Th("dewey"),
        html.Th("themes"),
        ]))
]
"""
HTML
"""
layout = html.Div(children=[
    html.H1(children='UAQUE: Generar recomendaciones'),

    html.Div(
        [
            dbc.Button("Load", id="loading-button", n_clicks=0),
            dbc.Spinner(html.Div(id="Agrupamiento")),
            dbc.Spinner(html.Div(id="Entrenamiento")),


        ]
    ),

])


@app.callback(
    Output("Agrupamiento", "children"),
    [Input("loading-button", "n_clicks")]
)
def load_output(n):

    if n:
        r = requests.get(url=url_agrupamiento, params={})
        print('Hecho!')

        return 'Hecho!'
@app.callback(
    Output("Entrenamiento", "children"),
    [Input("loading-button", "n_clicks")]
)
def load_output(n):

    if n:
        r = requests.get(url=url_entrenamient, params={})
        print('Hecho!')

        return 'Hecho!'
