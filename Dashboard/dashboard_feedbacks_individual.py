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

app = dash.Dash(__name__)
"""
Importacion de datos
"""

#Traemos los feedbacks de los usuarios con sus recomendaciones
feedbacks: pd.DataFrame = pd.DataFrame(pd.read_json('https://www.dropbox.com/s/fn2o86tbrplkjpd/recomedaciones_finalesMasFeedback.json?dl=1'))
feedbacks = feedbacks[['IDUsuario', 'Calificacion']]

reviewed_books = pd.DataFrame(feedbacks)

id_users = [{"label": x, "value": x } for x in reviewed_books["IDUsuario"].unique()]

"""
HTML
"""
app.layout = html.Div(children=[
    html.H1(children='UAQUE: Feedback de los usuarios individuales'),

    html.Div(
        dcc.Dropdown(
            id="users_id_dropdown",
            options=id_users,
            clearable=False,
            searchable=True,
            value="6466dbb15c41fdacb59eb1179817958de2c57191",
        )
    ),

    dcc.Graph(
        id='feedback_graph',
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
    #carga solo 50 resultados (no puede cargar los 40,000)
    return [o for o in id_users if search_value in o["label"]][0:50]

'''
Update graph based on
user list value
'''
@app.callback(
    Output("feedback_graph", "figure"),
    [Input("users_id_dropdown", "value")],
)

def update_graph(id_user):
    selected_row: pd.DataFrame  = reviewed_books.loc[(reviewed_books['IDUsuario'] == id_user) ]
    print(selected_row)
    scores = selected_row.groupby(['Calificacion']).size().reset_index(name='count')
    fig = px.pie(scores, values="count", names=['Dislike',  'No response','Like'])
    return fig

if __name__ == '__main__':
    print(reviewed_books)
    app.run_server(debug=False, port=8053)
