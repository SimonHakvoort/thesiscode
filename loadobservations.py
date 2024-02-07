import pandas as pd
import datetime


path_location = '/net/pc200002/nobackup/users/whan/WOW/KNMI_data/data/'
# filename = 


# df = pd.read_csv(path_location + filename)


# # Print first 5 rows of each column
# for column in df.columns:
#     print(f"{column}: {df[column].tail(5)}")

import os

# Get all csv files in path_location
csv_files = [file for file in os.listdir(path_location) if file.endswith('.csv')]
counter = 0
# Iterate over each csv file
for filename in csv_files:
    if filename != 'KIS_TOW_323_W_a_20160101-20161231.csv' and filename != 'KIS_TOW_229_W_a_20160101-20161231.csv':
        # Read the csv file
        df = pd.read_csv(path_location + filename)
        
        # Print the dimensions of the dataframe
        if df.shape[0] != 52561 and df.shape[0] != 52705 and df.shape[0] != 51265:
            print(f"{filename}: {df.shape}")
            counter += 1
print(counter)




def get_observation(data_frame, time_frame, data_transformer, start_time = '000000'):
    observations = []
    station_number = data_frame[10]['DS_CODE']
    station_name = data_frame[10]['DS_NAME']
    lattitude = data_frame[10]['DS_LAT']
    longtitude = data_frame[10]['DS_LON']



    first_index = 0
    start_date = datetime.datetime.strptime(data_frame['IT_DATETIME'][first_index], '%Y%m%d_%H%M%S_%f')




    
    
    return 


