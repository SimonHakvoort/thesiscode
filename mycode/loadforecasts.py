#!/usr/bin/env python3
import os
import pygrib
from forecast import Forecast
import pickle as pkl

def extract_from_grb(path, variable_name):
    grbs = pygrib.open(path)
                
    # Initialize variable to store the data
    variable = 0
    
    # Iterate through the messages (fields) in the GRIB file
    for grb in grbs:
        # Check if the current message corresponds to variable_name
        if variable_name in grb.parameterName:
            if not isinstance(variable, int):
                raise ValueError("Please enter a more precise variable name!")
            # Access the data values for the u-component and store them
            variable = grb.values

    if isinstance(variable, int):
        raise ValueError("Please enter a correct variable name!")

    # Close the GRIB file
    grbs.close()
    
    return variable

def get_filepath(year, month, day, initial_time, lead_time):
    if year == '2015' or year == '2016' or year == '2017':
        filename_pattern = f'HA40_N25_WINDFC_{year}{int(month):02d}{int(day):02d}{initial_time}_0{lead_time}00_GB'
        file_path = os.path.join('/net/pc230023/nobackup/users/ameronge/windwinter/output/', filename_pattern)
    
    elif year == '2018' or year == '2019':
        filename_pattern = f'HA40_N25_WindVerificationset_{year}{int(month):02d}{int(day):02d}{initial_time}_0{lead_time}00_GB'
        file_path = os.path.join('/net/pc230023/nobackup/users/ameronge/windwinter/verificationset/', filename_pattern)

    else:
        raise ValueError("Year not in range 2015-2019")

    return file_path

def make_forecast(year, month, day, initial_time, lead_time, variable_names):
    file_path = get_filepath(year, month, day, initial_time, lead_time)
    if os.path.isfile(file_path):
        variables = {}
        for variable_name in variable_names:
            variable = extract_from_grb(file_path, variable_name)
            variables[variable_name] = variable
        
        return Forecast(f'{year}-{month}-{day}', initial_time, lead_time, **variables)
    else:
        raise ValueError("File does not exist!")

def validation_data(variable_names, initial_time = '0000', lead_time = '24'):
    forecasts = []
    for year in range(2018, 2020):
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    forecast = make_forecast(str(year), str(month), str(day), initial_time, lead_time, variable_names)
                    forecasts.append(forecast)
                except ValueError:
                    continue
    return forecasts

def fold1_data(variable_names, initial_time = '0000', lead_time = '24'):
    forecasts = []
    year = 2015
    for month in range(10, 13):
        for day in range(1, 32):
            try:
                forecast = make_forecast(str(year), str(month), str(day), initial_time, lead_time, variable_names)
                forecasts.append(forecast)
            except ValueError:
                continue
    
    year = 2016
    for month in range(1, 4):
        for day in range(1, 32):
            try:
                forecast = make_forecast(str(year), str(month), str(day), initial_time, lead_time, variable_names)
                forecasts.append(forecast)
            except ValueError:
                continue

    return forecasts

def fold2_data(variable_names, initial_time = '0000', lead_time = '24'):
    forecasts = []
    year = 2016
    for month in range(10, 13):
        for day in range(1, 32):
            try:
                forecast = make_forecast(str(year), str(month), str(day), initial_time, lead_time, variable_names)
                forecasts.append(forecast)
            except ValueError:
                continue
    
    year = 2017
    for month in range(1, 4):
        for day in range(1, 32):
            try:
                forecast = make_forecast(str(year), str(month), str(day), initial_time, lead_time, variable_names)
                forecasts.append(forecast)
            except ValueError:
                continue

    return forecasts

def fold3_data(variable_names, initial_time = '0000', lead_time = '24'):
    forecasts = []
    year = 2017
    for month in range(10, 13):
        for day in range(1, 32):
            try:
                forecast = make_forecast(str(year), str(month), str(day), initial_time, lead_time, variable_names)
                forecasts.append(forecast)
            except ValueError:
                continue

    year = 2015
    for month in range(1, 4):
        for day in range(1, 32):
            try:
                forecast = make_forecast(str(year), str(month), str(day), initial_time, lead_time, variable_names)
                forecasts.append(forecast)
            except ValueError:
                continue
    
    return forecasts


def pickle_fold_forecasts(variable_names, i):
    nobackup = '/net/pc200239/nobackup/users/hakvoort/'

    if i == 1:
        forecasts = fold1_data(variable_names)
        folder_base = 'fold1data'
    elif i == 2:
        forecasts = fold2_data(variable_names)
        folder_base = 'fold2data'
    elif i == 3:
        forecasts = fold3_data(variable_names)
        folder_base = 'fold3data'
    elif i == 0:
        forecasts = validation_data(variable_names)
        folder_base = 'fold0data'
    else:
        raise ValueError("Invalid value for i. Expected values: 0, 1, 2, or 3.")

    counter = 0
    folder = os.path.join(nobackup, folder_base)
    while os.path.exists(folder):
        counter += 1
        folder = os.path.join(nobackup, f"{folder_base}_{counter}")
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, forecast in enumerate(forecasts):
        date = forecast.date.date()
        lead_time = forecast.lead_time.total_seconds() // 3600
        filename = f"{date}_{lead_time}.pkl"
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            pkl.dump(forecast, f)






