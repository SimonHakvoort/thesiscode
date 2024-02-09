import os
import numpy as np
import pandas as pd
import pickle as pkl
import datetime as dt

# Observation folder with raw CSV files 
csv_folder = '/net/pc200002/nobackup/users/whan/WOW/KNMI_data/data/'
# Target folder with sample Pickle files (NWP forecasts already extracted, run for train and test data)
nobackup = '/net/pc200239/nobackup/users/hakvoort/'
data_folder = nobackup + 'try/'

# Get .csv file names, .pkl file dates and station info
csv_files = sorted(os.listdir(csv_folder))
pkl_dates = {file : dt.datetime.strptime(file[:-4],'%Y%m%d') for file in os.listdir(data_folder)}
with open(nobackup + 'st-info.pkl', 'rb') as file:
    st_info = pkl.load(file)

# Loop over CSV files and insert relevant data to a corresponding Pickle file
for csv_file in csv_files:
    
    # Each CSV file contains observations for a given location (station) and a given range of dates
    # Get station code and range of dates for current CSV file     
    splits = csv_file.split('_')
    code = splits[2]
    start = dt.datetime.strptime(splits[-1][:8], '%Y%m%d') - dt.timedelta(days=1)
    end = dt.datetime.strptime(splits[-1][9:17], '%Y%m%d') + dt.timedelta(days=1)
    
    # Get corresponding Pickle files that fit in this date range     
    # Skip CSV file if no Pickle files in this date range
    pkl_files = [
        file for file, date in pkl_dates.items() if (start <= date) and (date <= end)
    ]
    if not pkl_files:
        continue
    
    # Group data in hourly observations
    df = pd.read_csv(csv_folder + csv_file)        
    df = df.rename(columns={'IT_DATETIME':'DATETIME', 'FF_10M_10':'OBS'}, inplace=False)
    df = df[['DATETIME', 'OBS']]
    df['DATETIME'] = pd.to_datetime(df['DATETIME'].str.slice(stop=11), format='%Y%m%d_%H')
    df = df.groupby('DATETIME').first()
    
    # Save observations to Pickle files
    for pkl_file in pkl_files:
        tag = pkl_file[:8] + '_' + code
        lead00 = pkl_dates[pkl_file]
        lead48 = lead00 + dt.timedelta(days=2)
        obs = df.loc[(lead00 <= df.index) & (df.index <= lead48)]
        index = list(map(lambda d: int(d.total_seconds() // 3600), obs.index - lead00))
        obs = obs.values[:,0]
        with open(data_folder + pkl_file, 'rb') as f:
            sample = pkl.load(f)
        sample[tag]['OBS'][index] = obs 
        with open(data_folder + pkl_file, 'wb') as f:
            pkl.dump(sample, f)


    print(f"Processed {csv_file} with {len(pkl_files)} samples")