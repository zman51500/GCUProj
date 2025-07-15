#This application creates a F1 data display that allows users to select races, drivers, and differnt forms of data analysis tools

import fastf1 as f1
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table, exceptions
import plotly.express as px
from plotly import io as pio
import io

app = Dash(__name__)

server = app.server

# Load event schedule and session data
f1.Cache.enable_cache('/Users/zanderbonnet/Desktop/GCU/Proj/f1_cache')
events = f1.get_event_schedule(2025, include_testing=False)
# Filter events to only include those that have occurred
comp = events.iloc[np.where(
    pd.to_datetime('today', utc=True) > pd.to_datetime(events['Session5DateUtc'], utc=True)
)]['EventName']
# Get the first event in the list
session = f1.get_session(2025, 'Australia', 'R')
session.load(telemetry=False, weather=False)
data = session.results[['ClassifiedPosition','Abbreviation','DriverNumber', 'Points']]

# Define Dash app layout
app.layout = html.Div(
    style={'backgroundColor': 'black', 'color': '#7FDBFF', 'margin': '0', 'padding': '0'},
    className='container-fluid',
    id='main-container',
    children=[

        html.Div(
            [
                # Help button
            html.Button(
                "Help",
                id="help-button",
                n_clicks=0,
                style={
                    'position': 'absolute',
                    'top': '35px',
                    'right': '10px',
                    'backgroundColor': '#7FDBFF',
                    'color': '#111111',
                    'border': 'none',
                    'padding': '10px 20px',
                    'cursor': 'pointer',
                    'zIndex': 1000
                }
            ),
                # Hidden help text
            dcc.Markdown(
                id="help-text",
                children="",
                style={
                    'display': 'none',
                    'margin': '20px',
                    'backgroundColor': '#222222',
                    'padding': '10px',
                    'borderRadius': '5px',
                    'color': '#7FDBFF'
                }
            )],
            style={'textAlign': 'center'}
        ),
        #Header
        html.H1(
            children='2025 F1 Race Analysis',
            className='header-title',
            style={'textAlign': 'center', 'color': '#7FDBFF'}
        ),
        #Creates object to store the data
        dcc.Store(id='stored-data', storage_type='memory'),
        html.Div(
            [
                # Create a list for selecting Race
                dcc.Markdown(children='Select Race to display.'),
                #Displays events that have occured
                dcc.RadioItems(
                    id='Race',
                    options=[{'label': event, 'value': event} for event in comp],
                    value=comp.iloc[0],
                    inline=True,
                )
            ],
            style={
                'width': '100%',
                'display': 'inline-block',
                'textAlign': 'center',
                'backgroundColor': '#111111',
                'color': '#7FDBFF'
            }
        ),
        #Creates place chart object
        html.Div(id = 'place_chart',
        children = [
            html.H2(f" Race Data for {session.event['EventName']}", style={'textAlign': 'center'}),
            dash_table.DataTable(
                id='raw-data-table',
                data=data.to_dict('records'),
                columns=[{"name": i, "id": i} for i in data.columns],
                fixed_rows={'headers': True},
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': 'black', 'fontWeight': 'bold'},
                style_data={'whiteSpace': 'normal', 'height': 'auto', 'width': 'auto'},
                page_size= 10,
            ),
        ]),
        html.Div([
            html.Button(
                "Save Table as CSV",
                id="save-csv-button",
                n_clicks=0,
                style={
                'margin': '20px auto',
                'display': 'block',
                'backgroundColor': '#7FDBFF',
                'color': '#111111',
                'border': 'none',
                'padding': '10px 20px',
                'cursor': 'pointer'
            }),
            dcc.Download(id = 'download-place'),
            html.Button(
                "Save Raw Data as CSV",
                id="save-og-button",
                n_clicks=0,
                style={
                'margin': '20px auto',
                'display': 'block',
                'backgroundColor': '#7FDBFF',
                'color': '#111111',
                'border': 'none',
                'padding': '10px 20px',
                'cursor': 'pointer'
            }),
            dcc.Download(id = 'download-raw')
        ],
            style= {'display':'flex'}
        ),

        dcc.Markdown(
            children='Select the drivers and method to display',
            style={'textAlign': 'center'}
        ),
        html.Div(
            [
                # Create a dropdown for selecting Drivers
                dcc.Dropdown(
                    id="Driver",
                    options=[driver for driver in session.laps['Driver'].unique()],
                    placeholder='Select a driver',
                    multi=True,
                    style={
                        'width': '100%',
                        'textAlign': 'center',
                        'backgroundColor': '#111111',
                        'color': 'black'
                    },
                    className='justify-content-center',
                    clearable=True,
                    searchable=True,
                    maxHeight=200
                ),
                dcc.Dropdown(
                    id="Method",
                    options=[
                        {'label': 'Place Chart', 'value': 'place_chart'},
                        {'label': 'Lap Time Box Plot', 'value': 'box_chart'},
                        {'label': 'Lap Time Chart', 'value': 'lap_time_chart'}
                    ],
                    value='place_chart',
                    placeholder='Select a method',
                    multi=False,
                    style={
                        'width': '100%',
                        'textAlign': 'center',
                        'backgroundColor': '#111111',
                        'color': 'black',
                        'textcolor': 'white'
                    },
                    className='justify-content-center',
                    clearable=False,
                    searchable=False,
                    maxHeight=200
                )
            ],
            style={
                'display': 'inline-block',
                'width': '100%',
                'textAlign': 'center',
                'backgroundColor': '#111111',
                'color': 'black'
            },
            className='justify-content-center'
        ),

        #plotly graph
        html.Div([
            dcc.Graph(
                id="Plot",
                figure={},
                style={
                'height': '70vh',
                'width': '100%',
                'textAlign': 'center',
                'backgroundColor': '#111111',
                'color': '#7FDBFF'
                }
            ),
        ], style={'textAlign': 'center'}),
        html.Div(id="save-pdf-status", style={'textAlign': 'center', 'color': '#7FDBFF', 'padding' : '100px'}, )
    ]
)

# Define callback to update the stored data
@app.callback(
    Output(component_id="stored-data", component_property="data"),
    Input(component_id = 'Race', component_property="value"),
)
def update_data(race):
    #If not race selcted do nothing
    if not race:
        raise exceptions.PreventUpdate
    # Load the session data for the race
    session = f1.get_session(2025, race, 'R')
    session.load(telemetry=False, weather=False)
    #convert to JSON
    df = {'res':session.results[['ClassifiedPosition','Abbreviation','DriverNumber', 'Points']].to_json(orient='split'),
          'lap': session.laps.to_json(orient='split'),
          'event': session.event['EventName']}
    return df


#Define callback for updating driver list
@app.callback(
    Output(component_id="Driver", component_property="options"),
    Input(component_id='stored-data', component_property="data"),
)
def update_drivers(race_data):
    if not race_data:
        raise exceptions.PreventUpdate
    drivers = pd.read_json(io.StringIO(race_data['lap']), orient='split')
    # Load the session data for the selected race
    # Update the driver options based on race selection
    return [driver for driver in drivers['Driver'].unique()]

#Callback for place table
@app.callback(
    Output(component_id="place_chart", component_property="children"),
    Input(component_id = 'stored-data', component_property="data")
)
def update_place_table(race_data):
    if not race_data:
        raise exceptions.PreventUpdate
    #read in data
    data = pd.read_json(io.StringIO(race_data['res']), orient='split')
    laps = pd.read_json(io.StringIO(race_data['lap'],), orient = 'split')
    #Find fastest lap for each driver and format data
    laps = (laps.groupby('Driver')['LapTime'].min()/1000)
    laps = pd.DataFrame(laps.values, index = laps.index)
    laps.columns = ['FastestLap (s)']
    laps = laps.fillna('No Laps Completed')
    #add fastest lap to dataframe
    data = data.join(laps, on = 'Abbreviation')
    #get event name
    e = race_data['event']
    #Create new data table with updated data
    return html.Div([
        html.H2(f" Race Data for {e}", style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='raw-data-table',
            data=data.to_dict('records'),
            columns=[{"name": i, "id": i} for i in data.columns],
            #sort_action='native',
            fixed_rows={'headers': True},
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': 'black', 'fontWeight': 'bold'},
            style_data={'whiteSpace': 'normal', 'height': 'auto', 'width': 'auto'},
            page_size= 10
        )
    ])
    
    # Load the session data for the

# Define callback to update the graph
@app.callback(
    Output(component_id="Plot", component_property="figure"),
    Input(component_id="Driver", component_property="value"),
    Input(component_id="Method", component_property="value"),
    Input(component_id = 'stored-data', component_property="data")
)
def update_graph(drivers, method, race_data):
    if not race_data:
        raise exceptions.PreventUpdate
    # Load the session data for the race selected
    event = race_data['event']
    session = pd.read_json(io.StringIO(race_data['lap']), orient='split')
    # If no drivers are selected, return a blank figure
    if not drivers:
        return gen_blank_fig(session)
    #filter by the selected drivers
    filtind = [index for index, value in enumerate(session['Driver']) if value in drivers]
    filt = session.iloc[filtind]
    #Selct what chart to generate based on method selcted

    if method == 'place_chart':
        return gen_place_chart(session, filt, event)
    elif method == 'box_chart':
        return box_chart(session, filt, event)
    elif method == 'lap_time_chart':
        return gen_lap_time_chart(session, filt, event)
    # If no method is selected, return a blank figure
    return gen_blank_fig(session)

# Generate a blank figure
def gen_blank_fig(session):
    fig = px.scatter(title="No drivers selected")
    fig.update_layout(
        xaxis_title='Lap Number',
        yaxis_title='Position',
        title_x=0.5,
        title_y=0.95,
        legend_title_text='Driver',
        paper_bgcolor='#111111',
        font_color='#7FDBFF'
    )
    fig.update_yaxes(
        dtick=1,
        range=[0, session['Position'].max()],
        title_standoff=10
    )
    fig.update_xaxes(
        dtick=1,
        range=[0, session['LapNumber'].max() + 1],
        title_standoff=10
    )
    return fig

# Generate the place chart
def gen_place_chart(ses, filt,e):
    fig = px.scatter(
        filt,
        x='LapNumber',
        y='Position',
        color='Driver',
        hover_name= 'Driver',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title=f"{e} F1 Place Chart"   
    )
    fig.update_traces(
        mode='lines+markers',
        marker=dict(size=5),
        line=dict(width=2)
    )
    fig.update_layout(
        xaxis_title='Lap Number',
        yaxis_title='Position',
        title_x=0.5,
        title_y=0.95,
        legend_title_text='Driver',
        paper_bgcolor='#111111',
        font_color='#7FDBFF'
    )
    fig.update_yaxes(
        dtick=1,
        range=[0, filt['Position'].max() + 1],
        title_standoff=10
    )
    fig.update_xaxes(
        dtick=1,
        range=[0, ses['LapNumber'].max() + 1],
        title_standoff=10
    )
    return fig

#generate box plot
def box_chart(ses, filt, e):
    filt['LapTime'] = filt['LapTime']/1000
    fig = px.box(
        filt,
        x='Driver',
        y='LapTime',
        color='Driver',
        hover_name= 'Driver',
        hover_data=['LapNumber', 'Compound'],
        points= 'all',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title=f"{e} F1 Lap Time Box Plot"
    )
    fig.update_layout(
        xaxis_title='Drivers',
        yaxis_title='LapTime (s)',
        title_x=0.5,
        title_y=0.95,
        legend_title_text='Driver',
        paper_bgcolor='#111111',
        font_color='#7FDBFF'
    )
    return fig
#generate lap times chart
def gen_lap_time_chart(ses, filt,e):
    filt['LapTime'] = filt['LapTime']/1000
    fig = px.scatter(
        filt,
        x='LapNumber',
        y='LapTime',
        color='Driver',
        hover_name= 'Driver',
        hover_data=['Compound'],
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title=f"{e} F1 Lap Time Chart"   
    )
    fig.update_traces(
        mode='lines+markers',
        marker=dict(size=5),
        line=dict(width=2)
    )
    fig.update_layout(
        xaxis_title='Lap Number',
        yaxis_title='LapTime (s)',
        title_x=0.5,
        title_y=0.95,
        legend_title_text='Driver',
        paper_bgcolor='#111111',
        font_color='#7FDBFF'
    )
    fig.update_yaxes(
        range=[filt['LapTime'].min() - 10, filt['LapTime'].max() + 10],
        title_standoff=10
    )
    fig.update_xaxes(
        range=[0, ses['LapNumber'].max() + 1],
        title_standoff=10
    )
    return fig

# Callback to toggle help text visibility
@app.callback(
    Output("help-text", "children"),
    Output("help-text", "style"),
    Input("help-button", "n_clicks")
)
def toggle_help(n_clicks):
    if n_clicks % 2 == 1:  # Show help text on odd clicks
        help_text = """
        **How to Use the Application:**
        - **Select Race:** Use the radio buttons to select a race from the list of completed events.
        - **Select Drivers:** Use the dropdown to select one or more drivers to display their positions over laps.
        - **Select Method:** Choose the method to display the data. Options include:
            - **Place Chart:** Displays the position of selected drivers over laps.
            - **Lap Time Box Plot:** Displays a box plot of lap times for selected drivers.
            - **Lap Time Chart:** Displays a scatter plot of lap times over laps for selected drivers.
        - **View Chart:** The chart will update automatically based on your selections.
        - **Save as PDF:** Click the "Save as PDF" button to download the current chart as a PDF file.
        - **Help:** Click the "Help" button to toggle this help text.
        - **Note:** Ensure you have selected at least one driver to view the charts.
        - **Disclaimer:** This application is for educational purposes only and does not represent official F1 data.
        """
        return help_text, {'backgroundColor': "#000000", 'padding': '0px', 'borderRadius': '5px', 'color': '#7FDBFF', 'display': 'block',
                           'textAlign': 'left'}
    else:  # Hide help text on even clicks
        return "", {'display': 'none'}
    
@app.callback(
    Output("download-place", "data"),
    Input("save-csv-button", "n_clicks"),
    Input("raw-data-table", "data")
)
def save_table_as_csv(n_clicks, data):
    df = pd.DataFrame(data)
    if n_clicks % 2 == 1:
        return dcc.send_data_frame(df.to_csv, 'Race_Data.csv')
    
@app.callback(
    Output('download-raw', 'data'),
    Input('save-og-button', 'n_clicks'),
    Input('stored-data', 'data')
)
def save_racedate(n_clicks, race_data):
    session = pd.read_json(io.StringIO(race_data['lap']), orient='split')
    if n_clicks % 2 == 1:
        return dcc.send_data_frame(session.to_csv, 'Raw_Race_Data.csv')

if __name__ == "__main__":
    app.run(debug=True)