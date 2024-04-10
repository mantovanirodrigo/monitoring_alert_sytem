
from sklearn.ensemble import IsolationForest

def get_IF_params(n_estimators = 100, contamination = 'auto'):
    
    params = {'n_estimators': 100,
              'contamination': 'auto'}
    
    return params    

def isolation_forest(df, feature, params):
    
    model = IsolationForest(**params)
    features = ['time_of_day','time_bin', feature]
    model.fit(df[features])
    predictions = model.predict(df[features])
    scores = model.decision_function(df[features])
    
    return model, predictions, scores
    