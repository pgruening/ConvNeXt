import pandas as pd  # (version 1.0.0)
import plotly  # (version 4.5.0)
import plotly.express as px

import dash  # (version 1.8.0)
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from DLBio.helpers import MyDataFrame, check_mkdir, search_rgx
from my_helpers import load_log
app = dash.Dash(__name__)

# ---------------------------------------------------------------
DATA_FOLDER = 'imagenet_exp/exp_data/imagenet'
RGX = r'(baseline|[01_]+)'


def get_data():
    folders_ = search_rgx(RGX, DATA_FOLDER)
    assert folders_
    logs_ = {}
    num_eps = 10000
    for folder in folders_:
        logs_[folder] = load_log(folder, DATA_FOLDER)
        keys_ = list(logs_[folder].keys())

    df = MyDataFrame()
    for name, values in logs_.items():
        tmp = {'name': name}
        num_entries = len(values[keys_[0]])
        for i in range(num_entries):
            for key in keys_:
                tmp[key] = values[key][i]
            df.update(tmp)

    return df.get_df(), keys_


df, dd_options = get_data()
xxx = 0

# ---------------------------------------------------------------
app.layout = html.Div([

    html.Div([
        dcc.Graph(id='our_graph')
    ], className='nine columns'),

    html.Div([

        html.Br(),
        html.Label(['Plot keys:'], style={
                   'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(id='drop_down',
                     options=dd_options,
                     value=dd_options[0],
                     multi=False,
                     disabled=False,
                     clearable=True,
                     searchable=True,
                     placeholder='Choose Plot key...',
                     className='form-dropdown',
                     style={'width': "90%"},
                     persistence='string',
                     persistence_type='memory'),

    ], className='three columns'),

])

# ---------------------------------------------------------------


@app.callback(
    Output('our_graph', 'figure'),
    [Input('drop_down', 'value')]
)
def build_graph(drop_down):
    return px.line(df, x="epoch", y=drop_down, color="name", height=600)

# ---------------------------------------------------------------


if __name__ == '__main__':
    app.run_server(debug=False)
