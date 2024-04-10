# -*- coding: utf-8 -*-

def data_prep(df, resample = 15, ratio = True, time_bins = 6, date = '2023-01-01'):
    
    import pandas as pd
    
    df = df.pivot_table(index='time',
                        columns='status',
                        values='count',
                        fill_value=0)
    df['reversed'] = df['reversed'] + df['backend_reversed']
    
    original_columns = df.columns.tolist()
    new_order = ['hour','minute','time_of_day'] + \
                df.columns.tolist() + \
                ['total']    
    df['total'] = df.sum(axis = 1)
    
    df['datetime'] = pd.to_datetime(f'{date} ' + df.index.str.replace('h ', ':'))
    df.set_index('datetime', inplace=True)    
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='T')
    df = df.reindex(full_range)
    df = df.fillna(0)
    print(df.columns)
    
    df['hour'] = df.index.hour.astype(int)
    df['minute'] = df.index.minute.astype(int)
    df['time_of_day'] = df['hour']*60 + df['minute']
    df = df[new_order]  

    
                   
    df.drop('backend_reversed', axis=1, inplace=True)
    
    bin_edges = pd.date_range(df.index.min(), df.index.max() + pd.Timedelta(minutes=1),
                              periods=time_bins+1)
    labels = [(i+1) for i in range(time_bins)]
    
    resample_dict = {}
    
    status_columns = ['approved', 'denied', 'failed',
           'processing', 'refunded', 'reversed', 'total']
    time_columns = ['hour','minute','time_of_day']
    
    
    if resample != False:        
    
        for col in df.columns:
            
            if col in time_columns:
                resample_dict[col] = 'max'
            elif col in status_columns:
                resample_dict[col] = 'sum'
        
        df = df.resample(f'{resample}T').agg(resample_dict)  
    
    df = df.fillna(0)    
                
    df['time_bin'] = pd.cut(df.index, bins = bin_edges, labels = labels,
                            include_lowest = True, right = False)   

    if ratio:
        for col in status_columns:
            if col != 'total':
                df.loc[df['total'] != 0, col] = df[col] / df['total']
            
    return df

def detection_plots(df, detection, predictions, scores):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patheffects as path_effects

    for status in detection:
          
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10,5))
        
        colors = np.where(predictions[status] == -1, 'r', 'g')
        
        ax.scatter(df.index, scores[status], c = colors, s = 14.0)
        ax.plot(df.index, scores[status], c = 'gold', lw = 0.7)    
        
        ax.set_ylabel('Anomaly score', fontsize = 12, weight = 'bold', c = 'gold')
        text = ax.yaxis.label
        text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])
        
        ax2 = ax.twinx()
        ax2.plot(df.index, df[status], c = 'white', lw = 1)
        ax2.set_ylabel(f'% {status}', fontsize = 12, c = 'white')
        text2 = ax2.yaxis.label
        text2.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])
        
        ax.set_facecolor('black')
        
        plt.show()
        
        

