# -*- coding: utf-8 -*-

import json
import pandas as pd
from email_alert_system import send_email
from datetime import timedelta
from joblib import load
import plotly.graph_objs as go
from project_utils import data_prep
from datetime import datetime

""" 

Table Updates 

"""

def update_failed_anomalies_table(data):
    if data is None:
        return []
    flags = json.loads(data)
    return flags.get('failed', [])

def update_reversed_anomalies_table(data):
    if data is None:
        return []
    flags = json.loads(data)
    return flags.get('reversed', [])

def update_denied_anomalies_table(data):
    if data is None:
        return []
    flags = json.loads(data)
    return flags.get('denied', [])

"""

Data for tables and storage

"""

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

def update_stored_data(n, current_time, models_dict, df, initial_time, end_time, stored_predicted_data, stored_accumulated_data):
    
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

"""

GRAPHS

"""

def update_graph_failed(stored_predicted_data, stored_accumulated_data, end_time):
    if stored_predicted_data and stored_accumulated_data:
        predicted_data = json.loads(stored_predicted_data)
        accumulated_data = json.loads(stored_accumulated_data)
        
        # Deserialize JSON to dataframes for the 'failed' feature
        predicted_df_failed = pd.read_json(predicted_data['failed'], orient='split')
        accumulated_df_failed = pd.read_json(accumulated_data['failed'], orient='split')
        
        return update_graph_live(predicted_df_failed, accumulated_df_failed, 'failed', end_time)
    return go.Figure()

def update_graph_reversed(stored_predicted_data, stored_accumulated_data, end_time):
    if stored_predicted_data and stored_accumulated_data:
        predicted_data = json.loads(stored_predicted_data)
        accumulated_data = json.loads(stored_accumulated_data)
        
        # Deserialize JSON to dataframes for the 'failed' feature
        predicted_df_failed = pd.read_json(predicted_data['reversed'], orient='split')
        accumulated_df_failed = pd.read_json(accumulated_data['reversed'], orient='split')
        
        return update_graph_live(predicted_df_failed, accumulated_df_failed, 'reversed', end_time)
    return go.Figure()

def update_graph_denied(stored_predicted_data, stored_accumulated_data, end_time):
    if stored_predicted_data and stored_accumulated_data:
        predicted_data = json.loads(stored_predicted_data)
        accumulated_data = json.loads(stored_accumulated_data)
        
        # Deserialize JSON to dataframes for the 'failed' feature
        predicted_df_failed = pd.read_json(predicted_data['denied'], orient='split')
        accumulated_df_failed = pd.read_json(accumulated_data['denied'], orient='split')
        
        return update_graph_live(predicted_df_failed, accumulated_df_failed, 'denied', end_time)
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
def update_graph_live(predicted_df, accumulated_df, feature, end_time):
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
        yaxis=dict(color='white', gridcolor='#2f2f2f'),
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



