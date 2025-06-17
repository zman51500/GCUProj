
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd

from strategy import simulate_strategy
from utils.fastf1_data import get_driver_pace, get_avg_pit_loss
from utils.recommender import recommend_strategy
from utils.export import export_to_json, export_to_pdf
from plotly.express import line

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def generate_stint_inputs(num_stints):
    return [
        html.Div([
            html.H5(f"Stint {i + 1}"),
            dcc.Dropdown(
                id={'type': 'tire-dropdown', 'index': i},
                options=[
                    {'label': 'Soft', 'value': 'Soft'},
                    {'label': 'Medium', 'value': 'Medium'},
                    {'label': 'Hard', 'value': 'Hard'},
                    {'label': 'Intermediate', 'value': 'Intermediate'},
                    {'label': 'Wet', 'value': 'Wet'}
                ],
                value='Medium'
            ),
            dcc.Input(
                id={'type': 'stint-laps', 'index': i},
                type='number',
                min=1,
                placeholder="Number of laps"
            )
        ]) for i in range(num_stints)
    ]

app.layout = dbc.Container([
    html.H2("F1 Tire Strategy Customizer"),
    dbc.Row([
        dbc.Col(dcc.Input(id='year', type='number', value=2024, placeholder="Year")),
        dbc.Col(dcc.Input(id='gp', type='text', value='Monaco', placeholder="Grand Prix")),
        dbc.Col(dcc.Input(id='driver', type='text', value='VER', placeholder="Driver Code")),
        dbc.Col(dcc.Input(id='laps', type='number', value=78, placeholder="Race Laps")),
        dbc.Col(dcc.Dropdown(id='weather', options=[{'label': i, 'value': i} for i in ['Dry', 'Mixed', 'Wet']], value='Dry', placeholder="Weather"))
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='recommend', options=[
            {'label': 'Yes', 'value': 'yes'},
            {'label': 'No', 'value': 'no'}
        ], value='yes', placeholder="Use recommended strategy?")),
        dbc.Col(dcc.Input(id='num-stints', type='number', min=1, max=5, value=3, placeholder="Number of stints"))
    ]),
    html.Div(id='stint-inputs'),
    html.Br(),
    dbc.Button("Run Strategy", id='run', color='primary'),
    html.Br(),
    html.Div(id='result'),
    dcc.Graph(id='lap-chart'),
    html.Br(),
    dbc.Button("Export as PDF", id='export-pdf'),
    html.Span(id='pdf-msg', style={"margin-left": "10px"}),
    html.Br(),
    dbc.Button("Export as JSON", id='export-json'),
    html.Span(id='json-msg', style={"margin-left": "10px"})
], fluid=True)

lap_data_store = []

@app.callback(
    Output('stint-inputs', 'children'),
    Input('num-stints', 'value')
)
def update_stint_inputs(num_stints):
    if num_stints is None or not isinstance(num_stints, int):
        return []
    return generate_stint_inputs(num_stints)

@app.callback(
    Output('result', 'children'),
    Output('lap-chart', 'figure'),
    Input('run', 'n_clicks'),
    State('year', 'value'),
    State('gp', 'value'),
    State('driver', 'value'),
    State('laps', 'value'),
    State('weather', 'value'),
    State('recommend', 'value'),
    State({'type': 'tire-dropdown', 'index': dash.ALL}, 'value'),
    State({'type': 'stint-laps', 'index': dash.ALL}, 'value')
)
def run_strategy(n, year, gp, driver, laps, weather, recommend, tire_vals, stint_lap_vals):
    if not n:
        return dash.no_update
    try:
        base_lap_time = get_driver_pace(year, gp, driver)
        pit_loss = get_avg_pit_loss(year, gp).mean().total_seconds()
    except Exception as e:
        return f"Error loading FastF1 data: {e}", {}
    if recommend == 'yes':
        strategy = recommend_strategy(laps, weather, year=year, gp=gp)
    else:
        strategy = [
            {"tire": tire_vals[i], "laps": stint_lap_vals[i]}
            for i in range(len(tire_vals))
            if tire_vals[i] and stint_lap_vals[i]
        ]
    total_time, lap_data = simulate_strategy(strategy, base_lap_time, pit_loss, total_laps=laps)
    global lap_data_store
    lap_data_store = lap_data
    df = pd.DataFrame(lap_data)
    fig = line(df, x='Lap', y='Time', color='Tire', title="Lap-by-Lap Race Time")
    return f"Total Estimated Race Time: {total_time:.2f} seconds", fig

@app.callback(
    Output('pdf-msg', 'children'),
    Input('export-pdf', 'n_clicks')
)
def export_pdf(n):
    if n:
        export_to_pdf(lap_data_store)
        return "📄 PDF Exported!"
    return ""

@app.callback(
    Output('json-msg', 'children'),
    Input('export-json', 'n_clicks')
)
def export_json(n):
    if n:
        export_to_json(lap_data_store)
        return "📁 JSON Exported!"
    return ""

if __name__ == "__main__":
    app.run(debug=True)
