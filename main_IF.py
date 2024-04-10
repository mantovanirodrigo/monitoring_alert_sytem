# -*- coding: utf-8 -*-

from project_utils import data_prep, detection_plots
import pandas as pd
from joblib import dump

# Importing files
file_no = 1
file = f'transactions_{file_no}'
raw_file = pd.read_csv(f'monitoring/{file}.csv')
raw_file2 = pd.read_csv('monitoring/transactions_2.csv')

df = data_prep(raw_file, resample = 5, ratio = True, time_bins = 6, date = '2023-01-01')
df2 = data_prep(raw_file2, resample = 5, ratio = True, time_bins = 6, date = '2023-01-02')

# Training isolation forest
from models import get_IF_params, isolation_forest

# Initial settings
detection = ['failed','reversed','denied'] # status features that we want to monitor
params = get_IF_params() # getting IF params

# Dicts for models, predictions and anomaly_scores
models = {}
predictions = {}
scores = {}

# Anomaly/outlier detection with Isolation Forest
for status in detection:
    
    model, prediction, anomaly_scores = isolation_forest(df = df,
                                                   feature = status, 
                                                   params = params)
    models[status] = model
    predictions[status] = prediction
    scores[status] = anomaly_scores

# Plotting detection    
detection_plots(df, detection, predictions, scores)
    
# Saving train_files with no outliers
for status in detection:
    
    df_normal = df[predictions[status] == 1]
    df_normal.to_csv(f'train_{status}.csv', index = True)
    
# Random Forest Regressor    
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Train and save RF models for failed, reversed and denied transactions
for status in detection:
    
    X = df.columns.tolist()
    X.remove(status)
    X.remove('minute')
    Y = status
    
    X_train, X_test, y_train, y_test = train_test_split(df_normal[X], df_normal[Y], test_size=0.2, random_state=42)
    model_RF = RandomForestRegressor(n_estimators=100, random_state=42)
    model_RF.fit(X_train, y_train)     

    predictions = model_RF.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    
    dump(model_RF, f'RandomForest/RF_{status}.joblib')
    
    



