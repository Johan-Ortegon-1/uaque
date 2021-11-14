from dash import dcc
from dash import html

layout = html.Div([
    # represents the URL bar, doesn't render anything
    dcc.Location(id='url', refresh=True),

    dcc.Link('Navigate to "/feedbacks por dewey"', href='/apps/feedbacks'),
    html.Br(),
    dcc.Link('Navigate to "/feedbacks individuales"', href='/apps/feedbackIndividual'),
    html.Br(),
    dcc.Link('Navigate to "/feedbacks individuales por dewey"', href='/apps/feedbackDewey'),
    html.Br(),
    dcc.Link('Navigate to "/grupos"', href='/apps/grupos'),
    html.Br(),
    dcc.Link('Navigate to "/pertenencias"', href='/apps/pertenencia'),
])

