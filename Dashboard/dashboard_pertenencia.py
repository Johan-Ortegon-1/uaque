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
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
"""
Importacion de datos
"""
#Casteamos el pd.read_json a un DataFrame


#traer tabla con: IDUsuario, DeweyUnidad, Titulo, Llave que fueron recomendados a los usuarios
recomendaciones_completas: pd.DataFrame = pd.DataFrame(pd.read_json('https://www.dropbox.com/s/0a9rcgbg8fmsbjt/recomedaciones_completas2.json?dl=1'))
#Traer pesos de los usuarios (suman 1)
pesos_usuario: pd.DataFrame = pd.DataFrame(pd.read_json('https://www.dropbox.com/s/ofoevb2xjp859vi/pesos_norm_id_unidad2.json?dl=1'))

"""
Creacion de labels para los dropdowns
"""
#Label: valor que se muestra en el dropdowns
#value: valor que tendra el dropdown despues de seleccionar
id_users = [{"label": x, "value": x } for x in pesos_usuario["IDUsuario"].unique()]

"""
HTML
"""
app.layout = html.Div(children=[
    html.H1(children='UAQUE: Pertenencia de los usuarios'),

    html.Div(
        dcc.Dropdown(
            id="users_id_dropdown",
            value="50052490d474975ef40a67220d0491571630eca6",
            clearable=False
        )
    ),

    html.Div(
        id="user_name",
        children=
        '''
        '''
    ),

    dcc.Graph(
        id='dewey_graph',
    ),
    html.Div(
        children=[
            html.Ul(id='book_list', children=[html.Li(i) for i in []]),
        ],
    ),

])

"""
Callbacks
Input: lo que cambiar. El primer valor es el id del item que cambia en el HTML. El segundo valor es child del item que cambia.
Output: item que cambia en reaccion. Lo mismo que arriba
Funcion debajo del callback es lo que controla el cambio. Las entradas de la funcion es el segundo valor
del input y el retorno es valor que va a tener el segundo argumento del output.
"""

#Cuando cambia el valor de busqueda, cambian las opciones que preesnta el dropdown.
@app.callback(
    Output("users_id_dropdown", "options"),
    Input("users_id_dropdown", "search_value")
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate
    #carga solo 50 resultados (no puede cargar los 40,000)
    return [o for o in id_users if search_value in o["label"]][0:50]

#Cuando cambia el valor del dropdown cambia el nombre de usuario del titulo
@app.callback(
    Output("user_name", "children"),
    [Input("users_id_dropdown", "value")]
)

def update_table_title(user_name):
    result =  '\n', str(user_name), '\n'
    return result

#Cuando cambia el valor del dropdown, cambia la lista de libros
@app.callback(
    Output("book_list", "children"),
    [Input("users_id_dropdown", "value")]
)
def update_book_list(dropdown_value):
    #Obtenemos la lista de libros recomendados del usuario
    does_user_id_match = recomendaciones_completas["IDUsuario"] == dropdown_value
    book_table = recomendaciones_completas.loc[does_user_id_match]
    book_table = book_table[['Titulo', 'Llave', 'DeweyUnidad']]
    book_list:list = []
    #Creamos la lista de listas, tal como se muestra en <li> de mozilla
    for i in range(len(book_table)):
        book_list.append(html.Li(book_table['Titulo'].values[i]))
        nested_list = html.Ul(children=[
            html.Li(
'Llave: ' + str(book_table['Llave'].values[i])
            ),
            html.Li(
                "Dewey: " + str(book_table['DeweyUnidad'].values[i])
            )
        ])
        book_list.append(nested_list)
    return book_list



#Cuando cambia el valor del dropdown, cambia la grafica
@app.callback(
    Output("dewey_graph", "figure"),
    [Input("users_id_dropdown", "value")]
)

def update_graph(dropdown_value):
    #seleccionamos filas pertinentes del usuario.
    selected_row = pesos_usuario.loc[pesos_usuario["IDUsuario"] == dropdown_value]
    #quitamos los ceros
    selected_row = selected_row.loc[:, (selected_row != 0.0).any(axis=0)]
    #multiplicamos los pesos de los usuarios *100
    selected_row = selected_row.apply(lambda x: x*100, axis=0)
    #quitamos el id del usuario
    selected_row = selected_row.drop(['IDUsuario'], axis=1)
    #Hacemos la fila a columna
    selected_row = selected_row.T

    fig = px.bar(selected_row, hover_data=['value'])
    #Estilo para la hoverinfo
    fig.update_traces(hovertemplate='<b>Pertenencia: %{y:.2f}%</b><extra></extra>')
    #Titulo de axis x
    fig.update_xaxes(type="category", title_text="Deweys")
    #Titulo de tabla
    fig.update_yaxes(title_text="Pertenencia (%)")
    #quitar legenda
    fig.update_layout(showlegend=False)
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
