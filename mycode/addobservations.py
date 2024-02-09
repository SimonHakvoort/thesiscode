import pickle as pkl
import os
import numpy as np
import pandas as pd
import datetime as dt

def addobservation(folder_pickle_files, csv_name):
    # example of forecast name is '2016-11-04_24.0.pkl'
    # example of csv file name is 'KIS_TOW_323_W_a_20160101-20161231.csv'

    csv_folder = '/net/pc200002/nobackup/users/whan/WOW/KNMI_data/data/'

    # Each CSV file contains observations for a given location (station) and a given range of dates
    # Get station code and range of dates for current CSV file     
    splits = csv_name.split('_')
    code = splits[2]
    start = dt.datetime.strptime(splits[-1][:8], '%Y%m%d') - dt.timedelta(days=1)
    end = dt.datetime.strptime(splits[-1][9:17], '%Y%m%d') + dt.timedelta(days=1)

    df = pd.read_csv(csv_folder + csv_name)
    df = df.rename(columns={'IT_DATETIME':'DATETIME', 'FF_10M_10':'OBS'}, inplace=False)
    df = df[['DATETIME', 'OBS']]
    df['DATETIME'] = pd.to_datetime(df['DATETIME'].str.slice(stop=13), format='%Y%m%d_%H%M')
    #df = df.groupby('DATETIME').first()

    for file in os.listdir(folder_pickle_files):
        if file.endswith('.pkl'):
            datestring = file[:10]
            pickle_date = dt.datetime.strptime(datestring, '%Y-%m-%d')
            if not start <= pickle_date <= end:
                continue


            with open(folder_pickle_files + file , 'rb') as f:
                sample = pkl.load(f)
            
            leadtime = int(file[11:13])
            forecasttime = pickle_date + dt.timedelta(hours=leadtime)
            rownumbers = np.where(df['DATETIME'] == forecasttime)[0]
            
            for index in rownumbers:
                if code in sample.observations and sample.observations[code][0] != df['OBS'][index]:
                    print('something wrong?')
                else:    
                    sample.add_observation(code, df['OBS'][index], forecasttime)
            #save sample again as pickle file
            with open(folder_pickle_files + file, 'wb') as f:
                pkl.dump(sample, f)

def addallobservations(pickle_folder):
    pickle_path = '/net/pc200239/nobackup/users/hakvoort/'
    csv_path = '/net/pc200002/nobackup/users/whan/WOW/KNMI_data/data/'
    for file in os.listdir(csv_path):
        addobservation(pickle_path + pickle_folder, file)
        #print("done with file: ", file)


