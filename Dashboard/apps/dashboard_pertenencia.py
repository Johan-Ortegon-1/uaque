# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
"""
Dashboard that shows user groups with percentages
and recommended books
"""
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import requests

import dash_bootstrap_components as dbc
from app import app

"""
Importacion de datos
"""
# Casteamos el pd.read_json a un DataFrame


"""
Creacion de labels para los dropdowns
"""
# Label: valor que se muestra en el dropdowns
# value: valor que tendra el dropdown despues de seleccionar

"""
HTML
"""
layout = html.Div(
    children=[
        html.H1(children="UAQUE: Pertenencia de los usuarios"),

        html.Div(
            dcc.Dropdown(
                id="users_id_pertenencia_dropdown",
                value="50052490d474975ef40a67220d0491571630eca6",
                clearable=False,
            )
        ),
        dbc.Spinner(children=[
                html.Div(
                    id="user_name",
                    children="""
                """,
                ),
                dcc.Graph(
                    id="dewey_graph",
                ),
                html.Div(
                    children=[
                        html.Ul(id="book_list", children=[html.Li(i) for i in []]),
                    ],
                ),
            ]),
    ]
)

"""
Callbacks
Input: lo que cambiar. El primer valor es el id del item que cambia en el HTML. El segundo valor es child del item que cambia.
Output: item que cambia en reaccion. Lo mismo que arriba
Funcion debajo del callback es lo que controla el cambio. Las entradas de la funcion es el segundo valor
del input y el retorno es valor que va a tener el segundo argumento del output.
"""

# Cuando cambia el valor de busqueda, cambian las opciones que preesnta el dropdown.
@app.callback(
    Output("users_id_pertenencia_dropdown", "options"),
    Input("users_id_pertenencia_dropdown", "search_value"),
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate
    # carga solo 50 resultados (no puede cargar los 40,000)

    smartuj_endpoint: str = "localhost:8000/api"
    uso_biblioteca: str = "suj-e-004"
    DashboardPertenenciaUpdateOptions: str = "DashboardPertenenciaUtilsUpdateOptions"

    url_pertenencia_update_options: str = (
        "http://"
        + smartuj_endpoint
        + "/"
        + uso_biblioteca
        + "/"
        + DashboardPertenenciaUpdateOptions
    )
    users_id = requests.get(
        url=url_pertenencia_update_options, params={"search_value": search_value}
        )
    users_id = users_id.json()
    return users_id


# Cuando cambia el valor del dropdown cambia el nombre de usuario del titulo
@app.callback(
    Output("user_name", "children"), [Input("users_id_pertenencia_dropdown", "value")]
)
def update_table_title(user_name):
    result = "\n", str(user_name), "\n"
    return result


# Cuando cambia el valor del dropdown, cambia la lista de libros
@app.callback(
    Output("book_list", "children"), [Input("users_id_pertenencia_dropdown", "value")]
)
def update_book_list(dropdown_value):

    smartuj_endpoint: str = "localhost:8000/api"
    uso_biblioteca: str = "suj-e-004"
    DashboardPertenenciaUpdateBookList: str = "DashboardPertenenciaUtilsUpdateBookList"

    url_pertenencia_update_book_list: str = (
        "http://"
        + smartuj_endpoint
        + "/"
        + uso_biblioteca
        + "/"
        + DashboardPertenenciaUpdateBookList
    )
    book_table = requests.get(
        url=url_pertenencia_update_book_list, params={"dropdown_value": dropdown_value}
    )
    book_table = pd.DataFrame.from_dict(book_table.json())
    book_list: list = []
    # Creamos la lista de listas, tal como se muestra en <li> de mozilla
    for i in range(len(book_table)):
        book_list.append(html.Li(book_table["Titulo"].values[i]))
        nested_list = html.Ul(
            children=[
                html.Li("Llave: " + str(book_table["Llave"].values[i])),
                html.Li("Dewey: " + str(book_table["DeweyUnidad"].values[i])),
            ]
        )
        book_list.append(nested_list)
    return book_list


# Cuando cambia el valor del dropdown, cambia la grafica
@app.callback(
    Output("dewey_graph", "figure"), [Input("users_id_pertenencia_dropdown", "value")]
)
def update_graph(dropdown_value):

    smartuj_endpoint: str = "localhost:8000/api"
    uso_biblioteca: str = "suj-e-004"
    dashboardGruposUtil: str = "DashboardPertenencia"

    # Agrupamiento crear perfiles grupales  http://{{smartuj-endpoint}}/{{perfil-grupal}}/model
    url_pertenencia: str = (
        "http://" + smartuj_endpoint + "/" + uso_biblioteca + "/" + dashboardGruposUtil
    )

    selected_row = requests.get(
        url=url_pertenencia, params={"dropdown_value": dropdown_value}
    )
    selected_row = selected_row.json()
    fig = px.bar(selected_row, hover_data=["value"])
    # Estilo para la hoverinfo
    fig.update_traces(hovertemplate="<b>Pertenencia: %{y:.2f}%</b><extra></extra>")
    # Titulo de axis x
    fig.update_xaxes(type="category", title_text="Deweys")
    # Titulo de tabla
    fig.update_yaxes(title_text="Pertenencia (%)")
    # quitar legenda
    fig.update_layout(showlegend=False)
    return fig
