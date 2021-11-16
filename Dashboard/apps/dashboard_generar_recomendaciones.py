import time
from dash.html.Col import Col from dash_bootstrap_components._components.Spinner import Spinner
import requests
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd

from app import app

import dash_bootstrap_components as dbc

"""
limpieza de datos

recibe url prestamos y material originales
path("suj-e-004/dataprep", DataPrepAPIView.as_view()),

Agrupamiento
dentro de perfil grupal
{perfil_grupal}/model

Entrenamiento
{recomendacion-uaque}/model

dentro de carpeta de perfil entrenamiento
dnetor de recomendar

"""
#limpieza de datos

smartuj_endpoint: str = 'localhost:8000/api'
recomendacion_uaque: str = 'suj-i-009'
perfil_grupal: str = 'suj-s-009'
uso_biblioteca: str = 'suj-e-004'
dataprep: str = ''

url_material: str= 'http://'+smartuj_endpoint+'/'+uso_biblioteca+'/'+'DashboardControlPanelMaterialUpdate'
url_prestamos: str= 'http://'+smartuj_endpoint+'/'+uso_biblioteca+'/'+'DashboardControlPanelPrestamosUpdate'

url_limpieza_datos: str= 'http://'+smartuj_endpoint+'/'+uso_biblioteca+'/'+'dataprep'

#Agrupamiento crear perfiles grupales  http://{{smartuj-endpoint}}/{{perfil-grupal}}/model
url_agrupamiento: str= 'http://'+smartuj_endpoint+'/'+perfil_grupal+'/model'
#Entrenar modelo generar recomendaciones http://{{smartuj-endpoint}}/{{recomendacion-uaque}}/model
url_entrenamiento: str= 'http://'+smartuj_endpoint+'/'+recomendacion_uaque+'/model'

"""
HTML
"""
layout = html.Div(children=[

    html.Div(
        [
            dbc.Container(children=[

                    html.H1(className="mt-10",children='UAQUE: Panel de control'),
                    dbc.Row(className="mt-2",children=
                        [
                            dbc.Col(width=3, children=[
                                dbc.Spinner(children=[
                                    dbc.Button("Actualizar URL Prestamos", id="actualizar_prestamos_button", n_clicks=0),
                                    ])
                                ]),
                            dbc.Col(width=3, children=[
                                dbc.Spinner(children=[
                                    dbc.Textarea(id="prestamos_url",placeholder='url')
                                    ])
                                ]),
                            dbc.Col(width=3,children=[
                                dbc.Spinner(html.Div(id="loading-output-prestamos")),
                                ]),
                        ]
                   ),
                    dbc.Row(className="mt-2",children=
                        [
                            dbc.Col(width=3, children=[
                                dbc.Spinner(children=[
                                    dbc.Button("Actualizar URL Material", id="actualizar_material_button", n_clicks=0),
                                    ])
                                ]),
                            dbc.Col(width=3, children=[
                                dbc.Spinner(children=[
                                    dbc.Textarea(id='material_url', placeholder='url')
                                    ])
                                ]),
                            dbc.Col(width=3,children=[
                                dbc.Spinner(html.Div(id="loading-output-material")),
                                ]),
                        ]
                   ),
                    dbc.Row(className="mt-2",children=
                        [
                            dbc.Col(width=6,children=[
                                dbc.Spinner(children=[
                                    dbc.Button("Iniciar limpieza de datos", id="limpieza-button", n_clicks=0),
                                    ])
                                ]),
                            dbc.Col(width=6,children=[
                                dbc.Spinner(html.Div(id="loading-output-limpieza")),
                                ]),
                        ]
                    ),
                    dbc.Row(className="mt-2",children=
                        [
                            dbc.Col(width=6,children=[
                                dbc.Spinner(children=[
                                    dbc.Button("Empezar agrupamiento", id="agrupamiento-button", n_clicks=0),
                                    ])
                                ]),
                            dbc.Col(width=6,children=[
                                dbc.Spinner(html.Div(id="loading-output-agrupamiento")),
                                ]),
                        ]
                    ),
                    dbc.Row(className="mt-2",children=
                        [
                            dbc.Col(width=6,children=[
                                dbc.Spinner(children=[
                                    dbc.Button("Empezar entrenamiento de modelo", id="entrenamiento-button", n_clicks=0),
                                    ])
                                ]),
                            dbc.Col(width=6,children=[
                                dbc.Spinner(html.Div(id="loading-output-entrenamiento")),
                                ]),
                        ]
                    ),
                ])


        ]
    ),

])
@app.callback(
    Output("loading-output-material", "children"),
   [Input("actualizar_material_button", "n_clicks")],
   [Input("material_url", "value")]
)
def request_material(n, nuevo_url_material):
    if n:
        r = requests.get(url=url_material, params={"url": nuevo_url_material})
        time.sleep(1)
        return "Material actualizado"

@app.callback(
    Output("loading-output-prestamos", "children"),
   [Input("actualizar_prestamos_button", "n_clicks")],
   [Input("material_url", "value")]
)
def request_prestamos(n, nuevo_url_prestamos):
    if n:
        r = requests.get(url=url_prestamos, params={"url": nuevo_url_prestamos})
        time.sleep(1)
        return "Prestamos actualizados"

@app.callback(
    Output("loading-output-limpieza", "children"),
   [Input("limpieza-button", "n_clicks")],
)
def request_limpieza(n):
    if n:
        r = requests.get(url=url_limpieza_datos, params={})
        time.sleep(1)
        return "Limpieza terminada"

@app.callback(
    Output("loading-output-agrupamiento", "children"),
   [Input("agrupamiento-button", "n_clicks")],
)
def request_agrupamiento(n):
    if n:
        r = requests.get(url=url_agrupamiento, params={})
        time.sleep(1)
        return "Agrupamiento terminado"

@app.callback(
    Output("loading-output-entrenamiento", "children"),
   [Input("entrenamiento-button", "n_clicks")],
)
def request_entrenamineto(n):
    if n:
        r = requests.get(url=url_entrenamiento, params={})
        time.sleep(1)
        return "Entrenamiento terminado"
