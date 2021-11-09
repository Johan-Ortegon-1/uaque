# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
'''
Dashboard that shows users in a particular
Dewey group
'''
import dash
from dash import dcc, html, dash_table
import pandas as pd
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
"""
Importacion de datos
"""
pesos_usuario: pd.DataFrame = pd.DataFrame(pd.read_json('https://www.dropbox.com/s/voqnwdzt8cwyr7u/pesos_norm_id_unidad.json?dl=1'))
join_tables: pd.DataFrame = pd.DataFrame(pd.read_json('https://www.dropbox.com/s/q38zr341seq7rkf/joinTablas.json?dl=1'))

table_columns = ['nombre_usuario', 'email','IDUsuario', 'Facultad', 'Programa']

all_deweys = join_tables[['DeweyUnidad', 'DeweyDecena', 'DeweyCentena']]
users_info = join_tables[['IDUsuario', 'Facultad', 'Programa','DeweyUnidad', 'DeweyDecena', 'DeweyCentena']]
users_info = users_info.drop_duplicates(subset=['IDUsuario', 'Facultad', 'Programa'])
users_info = users_info.merge(pesos_usuario, on='IDUsuario')
fake_users_info = pd.DataFrame(pd.read_json('https://www.dropbox.com/s/vb2uehwpmn2sboz/MOCK_DATA_ESTUDIANTES.json?dl=1'))

dewey_filters = [
        {"label":"Dewey Unidad", "value": "0.5"},
        {"label":"Dewey Decena", "value": "0.2"},
        {"label":"Dewey Centena","value": "0.1"},
        ]


app.layout = html.Div(children=[
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
            id="dewey_list_dropdown",
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
    Output("dewey_list_dropdown", "options"),
    Input("dewey_filter_dropdown", "value")
)
def update_dewey_list_options(selected_dewey_level):
    dewey_list = []
    selected_dewey_option = level_to_dewey_option(selected_dewey_level)

    dewey_list = all_deweys[selected_dewey_option].unique()
    dewey_list = [{"label": x, "value": x } for x in dewey_list]

    return dewey_list


@app.callback(
    Output("users_table", "data"),
    [Input("dewey_list_dropdown", "value")],
)

def update_table(dewey):
    threshold = 0.2
    if dewey == -999:
        dewey = "-999"
    does_dewey_match = users_info[str(dewey)] >= threshold
    selected_rows = users_info.loc[does_dewey_match]
    selected_rows = selected_rows[['IDUsuario', 'Facultad', 'Programa']]
    #insert mock data
    n_rows = len(selected_rows.index)
    if n_rows>len(fake_users_info.index):
        n_rows = len(fake_users_info.index)
    fake_info = fake_users_info.sample(n=n_rows)
    selected_rows = pd.concat([selected_rows.reset_index(),fake_info.reset_index()], axis=1)
    return selected_rows[table_columns].to_dict('records')

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


if __name__ == '__main__':
    app.run_server(debug=False, port=8052)
