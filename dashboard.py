# -*- coding: utf-8 -*-

# dash
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import dash_table

# auxiliary libs
import pandas as pd
import plotly.graph_objs as go
from datetime import timedelta
from joblib import load
from datetime import datetime
import json

# utils and aggregated systems
from project_utils import data_prep
from email_alert_system import send_email

# Reading test file
raw_file2 = pd.read_csv('monitoring/transactions_2.csv')
df = data_prep(raw_file2, resample = 5, ratio = True, time_bins = 6, date = '2023-01-02')

# Initiating app
app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])


# Initial setup
initial_time = df.index.min()
end_time = df.index.max()
current_time = initial_time
start_datetime = datetime.strptime('2023/01/02 00:00', '%Y/%m/%d %H:%M')
end_datetime = datetime.strptime('2023/01/03 00:00', '%Y/%m/%d %H:%M')
start_timestamp = int(start_datetime.timestamp())
end_timestamp = int(end_datetime.timestamp())

#  Setting up dictionaries for data and models
accumulated_data = {'failed':pd.DataFrame(columns = df.columns),
                    'reversed':pd.DataFrame(columns = df.columns),
                    'denied':pd.DataFrame(columns = df.columns)}
predicted_data = {'failed':pd.DataFrame(),
                  'reversed':pd.DataFrame(),
                  'denied':pd.DataFrame()}
models_dict = {}

# App Layout
app.layout = html.Div([
    
    # Store units
    dcc.Store(id='store-predicted-data'), # Predicted data
    dcc.Store(id='store-accumulated-data'), # Accumulated data
    dcc.Store(id='anomaly-flags-store'), # Anomaly flags
    
    # Interval
    dcc.Interval(
        id='interval-component',
        interval=4*1000,  # in milliseconds
        n_intervals=0
    ),

    # Dashboard header    
    html.Div(className='row', children=[
        html.Img(src='https://asset.brandfetch.io/idQCNoAdNh/idN7ZG6N9s.png', style={
            'width': '150px', 'height': 'auto', 'marginLeft': '50px', 'marginTop': '10px',
            'marginRight':'-200px', 'marginBottom':'-10px'
        }),
        html.H2(children='Anomaly Detection Dashboard', style={'color': 'white', 'marginTop':'10px'}),
        html.Div(style={'width': '20px'})
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'space-between', 'backgroundColor': '#3c41f5'}),
    html.H3(children='Transactions', style={'color': 'white', 'textAlign': 'center', 'backgroundColor': '#3c41f5', 'marginBottom':'0px'}),
    html.Hr(style={'border': '2px solid white', 'margin': '0px', 'backgroundColor':'#3c41f5'}),
    
    # Title for first row (plots)
    html.Div(className='container-fluid', style={'maxWidth': '120vw', 'fontSize': '8px'}, children=[
        html.Div(className='row', style={'flex': '1'}, children=[
            html.Div([html.H2(children='Failed')], className='small-height-col'),
            html.Div([html.H2(children='Reversed')], className='small-height-col'),
            html.Div([html.H2(children='Denied')], className='small-height-col')            
        ]),
        
        # Graph-row
        html.Div(className='row', style={'flex': '1'}, children=[
            html.Div([dcc.Graph(id='graph-failed')], id='container-failed', className='col-md-4', style={'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'}),
            html.Div([dcc.Graph(id='graph-reversed')], id='container-reversed', className='col-md-4', style={'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'}),
            html.Div([dcc.Graph(id='graph-denied')], id='container-denied', className='col-md-4', style={'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'})
        ]),
        
        # Detection warning boxes row
        html.Div(className='row', style={'flex': '1'}, children=[
            html.Div(id='warning-failed', className='warning-box', children=[
                html.Span(id='deviation-failed'),
                html.Span(id='icon-failed')
            ]),
            html.Div(id='warning-reversed', className='warning-box', children=[
                html.Span(id='deviation-reversed'),
                html.Span(id='icon-reversed')
            ]),
            html.Div(id='warning-denied', className='warning-box', children=[
                html.Span(id='deviation-denied'),
                html.Span(id='icon-denied')
            ])
        ]),
        
        # Tables that show anomaly flags
        html.Div(className='row', style={'flex':'1', 'fontsize':'8px'}, children=[
            # Headings in a single flex row
            html.Div(className = 'row', style = {'flex':'1', 'width':'100%'}, children=[
                html.Div([html.H3([html.Span('Flags for anomalies'), html.Br(), html.Span('in failed transactions')],
                                  style={'fontSize': '14px'})],className = 'small-height-col2'),
                html.Div([html.H3([html.Span('Flags for anomalies'), html.Br(), html.Span('in reversed transactions')],
                                   style={'fontSize': '14px'})], className = 'small-height-col2'),
                html.Div([html.H3([html.Span('Flags for anomalies'), html.Br(), html.Span('in denied transactions')],
                                   style={'fontSize': '14px'})], className = 'small-height-col2')
            ])
        ]),
        # DataTables below the headings
        html.Div(className='row', style={'flex':'1'}, children=[
            # DataTable for Failed Anomalies
            html.Div(className='table-container', children=[
                dash_table.DataTable(id='failed-anomalies-table',
                                     columns=[{'name': 'Timestamp', 'id': 'timestamp'},
                                              {'name': 'Predicted value', 'id': 'predicted_value'},
                                              {'name': 'Actual value', 'id': 'actual_value'}],
                                     style_table={'maxHeight': '300px', 'overflowY': 'scroll'},
                                     style_data={'textAlign': 'center', 'color': 'black'},
                                     style_header={'backgroundColor': 'black', 'color': 'white',
                                                   'fontWeight': 'bold', 'textAlign': 'center',
                                                   'fontSize':'12px'},
                                     style_cell={'fontSize': '14px'})
            ]),
            # DataTable for Reversed Anomalies
            html.Div(className='table-container', children=[
                dash_table.DataTable(id='reversed-anomalies-table',
                                     columns=[{'name': 'Timestamp', 'id': 'timestamp'},
                                              {'name': 'Predicted value', 'id': 'predicted_value'},
                                              {'name': 'Actual value', 'id': 'actual_value'}],
                                     style_table={'maxHeight': '300px', 'overflowY': 'scroll'},
                                     style_data={'textAlign': 'center', 'color': 'black'},
                                     style_header={'backgroundColor': 'black', 'color': 'white',
                                                   'fontWeight': 'bold', 'textAlign': 'center',
                                                   'fontSize':'12px'},
                                     style_cell={'fontSize': '14px'})
            ]),
            # DataTable for Denied Anomalies
            html.Div(className='table-container', children=[
                dash_table.DataTable(id='denied-anomalies-table',
                                     columns=[{'name': 'Timestamp', 'id': 'timestamp'},
                                              {'name': 'Predicted value', 'id': 'predicted_value'},
                                              {'name': 'Actual value', 'id': 'actual_value'}],
                                     style_table={'maxHeight': '300px', 'overflowY': 'scroll'},
                                     style_data={'textAlign': 'center', 'color': 'black'},
                                     style_header={'backgroundColor': 'black', 'color': 'white',
                                                   'fontWeight': 'bold', 'textAlign': 'center',
                                                   'fontSize':'12px'},
                                     style_cell={'fontSize': '14px'})
            ]),
        ])
    ])
])













"""

CALLBACKS

"""

# Update failed anomalies table
@app.callback(
    Output('failed-anomalies-table', 'data'),
    [Input('anomaly-flags-store', 'data')]
)
def update_failed_anomalies_table(data):
    if data is None:
        return []
    flags = json.loads(data)
    return flags.get('failed', [])

# Update reversed anomalies table
@app.callback(
    Output('reversed-anomalies-table', 'data'),
    [Input('anomaly-flags-store', 'data')]
)
def update_reversed_anomalies_table(data):
    if data is None:
        return []
    flags = json.loads(data)
    return flags.get('reversed', [])

# Update denied anomalies table
@app.callback(
    Output('denied-anomalies-table', 'data'),
    [Input('anomaly-flags-store', 'data')]
)
def update_denied_anomalies_table(data):
    if data is None:
        return []
    flags = json.loads(data)
    return flags.get('denied', [])

# Update anomaly_flags
@app.callback(
    Output('anomaly-flags-store', 'data'),
    [Input('interval-component', 'n_intervals')],
    [State('anomaly-flags-store', 'data'),
     State('store-predicted-data', 'data'),
     State('store-accumulated-data', 'data')]
)
def simulate_anomaly_detection(n, existing_flags_json, stored_predicted_data, stored_accumulated_data):
    # Placeholder for actual anomaly detection logic
    # Here, we're just simulating a new flag for each category for demonstration
    
    if existing_flags_json is None:
        existing_flags = {'failed': [], 'reversed': [], 'denied': []}
    else:
        existing_flags = json.loads(existing_flags_json)
    
    # Example for 'failed' feature
    if stored_predicted_data is None or stored_accumulated_data is None:
        return json.dumps(existing_flags)
    
    predicted_data = json.loads(stored_predicted_data)
    accumulated_data = json.loads(stored_accumulated_data)
    
    # Failed category
    predicted_df_failed = pd.read_json(predicted_data['failed'], orient='split')
    accumulated_df_failed = pd.read_json(accumulated_data['failed'], orient='split')
    
    # Calculate deviation for the latest data point
    latest_predicted_failed = predicted_df_failed['failed'].iloc[-1]
    latest_actual_failed = accumulated_df_failed['failed'].iloc[-1]
    deviation_failed = latest_actual_failed - latest_predicted_failed
    
    mean = {'failed':1.4265525643465073e-05,
            'reversed':0.007517223071899441,
            'denied':0.09771097246986074}
    std = {'failed':9.578193261812571e-05,
           'reversed':0.028901789468009087,
           'denied':0.031348114690779014}
   # Define your threshold
    threshold = {'failed':std['failed']*3,
                 'reversed':std['reversed']*3,
                 'denied':std['denied']*3}
    
    if deviation_failed > threshold['failed']:
        new_flag_failed = {'timestamp': str(accumulated_df_failed.index[-1]).split('+')[0],
                           'predicted_value': f'{(100*latest_predicted_failed).round(2)}%',
                           'actual_value': f'{(100*latest_actual_failed).round(2)}%'}
        existing_flags['failed'].append(new_flag_failed)
        
        send_email(feature = 'failed', timestamp = str(accumulated_df_failed.index[-1]).split('+')[0],
                    actual_value = latest_actual_failed, predicted_value = latest_predicted_failed,
                    threshold = threshold['failed'])
    
    # Reversed category
    predicted_df_reversed = pd.read_json(predicted_data['reversed'], orient='split')
    accumulated_df_reversed = pd.read_json(accumulated_data['reversed'], orient='split')
    
    # Calculate deviation for the latest data point
    latest_predicted_reversed = predicted_df_reversed['reversed'].iloc[-1]
    latest_actual_reversed = accumulated_df_reversed['reversed'].iloc[-1]
    deviation_reversed = latest_actual_reversed - latest_predicted_reversed
    
    if deviation_reversed > threshold['reversed']:
        new_flag_reversed = {'timestamp': str(accumulated_df_reversed.index[-1]).split('+')[0],
                             'predicted_value': f'{(100*latest_predicted_reversed).round(2)}%',
                             'actual_value': f'{(100*latest_actual_reversed).round(2)}%'}
        existing_flags['reversed'].append(new_flag_reversed)
        
        send_email(feature = 'reversed', timestamp = str(accumulated_df_reversed.index[-1]).split('+')[0],
                    actual_value = latest_actual_reversed, predicted_value = latest_predicted_reversed,
                    threshold = threshold['reversed'])
    
    # Denied category
    predicted_df_denied = pd.read_json(predicted_data['denied'], orient='split')
    accumulated_df_denied = pd.read_json(accumulated_data['denied'], orient='split')
    
    # Calculate deviation for the latest data point
    latest_predicted_denied = predicted_df_denied['denied'].iloc[-1]
    latest_actual_denied = accumulated_df_denied['denied'].iloc[-1]
    deviation_denied = latest_actual_denied - latest_predicted_denied
    
    if deviation_denied > threshold['denied']:
        new_flag_denied = {'timestamp': str(accumulated_df_denied.index[-1]).split('+')[0],
                           'predicted_value': f'{(100*latest_predicted_denied).round(2)}%',
                           'actual_value': f'{(100*latest_actual_denied).round(2)}%'}
        existing_flags['denied'].append(new_flag_denied)
        
        send_email(feature = 'denied', timestamp = str(accumulated_df_denied.index[-1]).split('+')[0],
                    actual_value = latest_actual_denied, predicted_value = latest_predicted_denied,
                    threshold = threshold['denied'])
    
    return json.dumps(existing_flags)

@app.callback(
    [Output('warning-failed', 'style'),
     Output('deviation-failed', 'children'),
     Output('icon-failed', 'children'),
     Output('container-failed', 'style')],
    [Input('store-predicted-data', 'data'),
     Input('store-accumulated-data', 'data')]
    )
def update_warning_message_failed(stored_predicted_data, stored_accumulated_data):
    
    return update_warning_message(stored_predicted_data, stored_accumulated_data, 'failed')

@app.callback(
    [Output('warning-reversed', 'style'),
     Output('deviation-reversed', 'children'),
     Output('icon-reversed', 'children'),
     Output('container-reversed', 'style')],
    [Input('store-predicted-data', 'data'),
     Input('store-accumulated-data', 'data')]
    )
def update_warning_message_reversed(stored_predicted_data, stored_accumulated_data):
    
    return update_warning_message(stored_predicted_data, stored_accumulated_data, 'reversed')

@app.callback(
    [Output('warning-denied', 'style'),
     Output('deviation-denied', 'children'),
     Output('icon-denied', 'children'),
     Output('container-denied', 'style')],
    [Input('store-predicted-data', 'data'),
     Input('store-accumulated-data', 'data')]
    )
def update_warning_message_denied(stored_predicted_data, stored_accumulated_data):
    
    return update_warning_message(stored_predicted_data, stored_accumulated_data, 'denied')

@app.callback(
    [Output('store-predicted-data', 'data'),
     Output('store-accumulated-data', 'data')],
    [Input('interval-component', 'n_intervals')],
    [State('store-predicted-data', 'data'),
     State('store-accumulated-data', 'data')]
)
def update_stored_data(n, stored_predicted_data, stored_accumulated_data):
    global current_time, models_dict, df, initial_time, end_time
    
    # Initialize or update from stored JSON
    if stored_predicted_data is not None and stored_accumulated_data is not None:
        predicted_data = {k: pd.read_json(v, orient='split') for k, v in json.loads(stored_predicted_data).items()}
        accumulated_data = {k: pd.read_json(v, orient='split') for k, v in json.loads(stored_accumulated_data).items()}
    else:
        predicted_data = {feature: pd.DataFrame() for feature in ['failed', 'reversed', 'denied']}
        accumulated_data = {feature: pd.DataFrame() for feature in ['failed', 'reversed', 'denied']}
    
    if current_time > end_time:
        current_time = initial_time  # Reset if we reach the end
    
    next_time = current_time + timedelta(minutes=5)
    slice_df = df.loc[current_time:next_time]
    
    for feature in ['failed', 'reversed', 'denied']:
        if feature not in models_dict:
            models_dict[feature] = load(f'RandomForest/RF_{feature}.joblib')
        
        x_columns = df.columns.tolist()
        x_columns.remove(feature)
        x_columns.remove('minute')  # Assuming 'minute' is a column to be removed
        
        feature_predicted = models_dict[feature].predict(slice_df[x_columns])
        predicted_slice_df = pd.DataFrame(feature_predicted, columns=[feature], index=slice_df.index)
        predicted_data[feature] = pd.concat([predicted_data[feature], predicted_slice_df])
        accumulated_data[feature] = pd.concat([accumulated_data[feature], slice_df[[feature]]])
    
    current_time = next_time
    
    # Convert updated dictionaries of dataframes to JSON for storage
    predicted_data_json = {k: v.to_json(date_format='iso', orient='split') for k, v in predicted_data.items()}
    accumulated_data_json = {k: v.to_json(date_format='iso', orient='split') for k, v in accumulated_data.items()}
    
    return json.dumps(predicted_data_json), json.dumps(accumulated_data_json)

# Updating Graph-failed
@app.callback(Output('graph-failed', 'figure'),
              [Input('store-predicted-data', 'data'),
               Input('store-accumulated-data', 'data')])
def update_graph_failed(stored_predicted_data, stored_accumulated_data):
    if stored_predicted_data and stored_accumulated_data:
        predicted_data = json.loads(stored_predicted_data)
        accumulated_data = json.loads(stored_accumulated_data)
        
        # Deserialize JSON to dataframes for the 'failed' feature
        predicted_df_failed = pd.read_json(predicted_data['failed'], orient='split')
        accumulated_df_failed = pd.read_json(accumulated_data['failed'], orient='split')
        
        return update_graph_live(predicted_df_failed, accumulated_df_failed, 'failed')
    return go.Figure()

# Updating graph-reversed
@app.callback(Output('graph-reversed', 'figure'),
              [Input('store-predicted-data', 'data'),
               Input('store-accumulated-data', 'data')])
def update_graph_reversed(stored_predicted_data, stored_accumulated_data):
    if stored_predicted_data and stored_accumulated_data:
        predicted_data = json.loads(stored_predicted_data)
        accumulated_data = json.loads(stored_accumulated_data)
        
        # Deserialize JSON to dataframes for the 'failed' feature
        predicted_df_failed = pd.read_json(predicted_data['reversed'], orient='split')
        accumulated_df_failed = pd.read_json(accumulated_data['reversed'], orient='split')
        
        return update_graph_live(predicted_df_failed, accumulated_df_failed, 'reversed')
    return go.Figure()

# Updating graph-denied
@app.callback(Output('graph-denied', 'figure'),
              [Input('store-predicted-data', 'data'),
               Input('store-accumulated-data', 'data')])
def update_graph_denied(stored_predicted_data, stored_accumulated_data):
    if stored_predicted_data and stored_accumulated_data:
        predicted_data = json.loads(stored_predicted_data)
        accumulated_data = json.loads(stored_accumulated_data)
        
        # Deserialize JSON to dataframes for the 'failed' feature
        predicted_df_failed = pd.read_json(predicted_data['denied'], orient='split')
        accumulated_df_failed = pd.read_json(accumulated_data['denied'], orient='split')
        
        return update_graph_live(predicted_df_failed, accumulated_df_failed, 'denied')
    return go.Figure()

"""
GENERAL FUNCTIONS
used in callbacks
"""
# Updating warnings
def update_warning_message(stored_predicted_data, stored_accumulated_data, feature):
    if stored_predicted_data is None or stored_accumulated_data is None:
        return {'display': 'none'}, '', ''
    
    # Example for 'failed' feature
    predicted_data = json.loads(stored_predicted_data)
    accumulated_data = json.loads(stored_accumulated_data)
    
    predicted_df_failed = pd.read_json(predicted_data[feature], orient='split')
    accumulated_df_failed = pd.read_json(accumulated_data[feature], orient='split')
    
    # Calculate deviation for the latest data point
    latest_predicted = predicted_df_failed[feature].iloc[-1]
    latest_actual = accumulated_df_failed[feature].iloc[-1]
    deviation = latest_actual - latest_predicted
    
    mean = {'failed':1.4265525643465073e-05,
            'reversed':0.007517223071899441,
            'denied':0.09771097246986074}
    std = {'failed':9.578193261812571e-05,
           'reversed':0.028901789468009087,
           'denied':0.031348114690779014}
    threshold = 3*std[feature]
    
    # Define your threshold
    
    if deviation > threshold:
        box_style = {'display': 'block', 'background-color': 'red', 'color': 'white', 'padding': '10px',
                     'width':'480px', 'height':'50px', 'textAlign':'center', 'font-size':'20px',
                     'box-shadow': '4px 4px 8px white','flex': '1'}
        icon = ' ⚠️'
        container_style = {'box-shadow': '4px -8px 8px red'}
    else:
        box_style = {'display': 'block', 'background-color': 'green', 'color': 'white', 'padding': '10px',
                     'width':'480px', 'height':'50px', 'textAlign':'center', 'font-size':'20px',
                     'box-shadow': '4px 4px 8px white','flex': '1'}
        icon = ' ✓'
        container_style = {'box-shadow': '4px -8px 8px green'}
    
    deviation_text = f"Deviation: {(100*deviation):.2f}% \n (threshold: {(threshold*100):.2f}%)"
    
    return box_style, deviation_text, icon, container_style
    
# Updating graph-general
def update_graph_live(predicted_df, accumulated_df, feature):
    """
    Update the live graph based on the feature specified.
    
    Parameters:
    - predicted_df: DataFrame containing predicted data for the specified feature.
    - accumulated_df: DataFrame containing accumulated actual data for the specified feature.
    - feature: The feature to plot (e.g., 'failed', 'reversed', 'denied').
    """
    
    # Create the figure
    figure = go.Figure(data=[
        go.Scatter(
            x=accumulated_df.index,
            y=accumulated_df[feature],  # Plotting the actual values for the feature
            mode='lines+markers',
            name='Actual',
            line=dict(color='red', width=1)  # Actual values in red
        ),
        go.Scatter(
            x=predicted_df.index,
            y=predicted_df[feature],  # Plotting the predicted values for the feature
            mode='lines',
            name='Predicted',
            line=dict(color='blue', width=2, dash='dash')  # Predicted values in blue, dashed line
        )
    ],
    layout=go.Layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        title_font=dict(color='white'),
        paper_bgcolor='black',
        plot_bgcolor='#1c1b1b',
        xaxis=dict(color='white', gridcolor='#2f2f2f'),
        yaxis=dict(color='white', gridcolor='#2f2f2f', tickformat=',.0%'),
        xaxis_title='Time',
        yaxis_title='%',
        font=dict(color='white'),
        height=250,
        width=450,
        margin=dict(l=0, r=0, t=30, b=0)
    ))
    
    if accumulated_df.index.tz_localize(None)[-1] > end_time:  # Check if current time exceeds end time
            return go.Figure()
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
