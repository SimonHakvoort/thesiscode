#!/usr/bin/env python3
import os
import pygrib
from src.loading_data.forecast import Forecast
import pickle as pkl


def ObtainParameterName(parameter_name):
    # returns a tuple, where the first element is the indicatorOfParameter for the grb and the second element is the level
    if parameter_name == 'u_wind10':
        return (33,10)
    elif parameter_name == 'v_wind10':
        return (34,10)
    elif parameter_name == 'press':
        return (1, 0)
    elif parameter_name == 'kinetic':
        return (200, 47)
    elif parameter_name =='humid':
        return (52, 2)
    elif parameter_name == 'geopot':
        return (6, 700)
    else:
        raise ValueError("Invalid parameter name")

def extract_from_grb(path, variable_name):
    grbs = pygrib.open(path)

    try:
        indicatorOfParameter, level = ObtainParameterName(variable_name)
        variable = grbs.select(indicatorOfParameter=indicatorOfParameter, level=level)
        return variable[0].values
    except:
        raise ValueError("Unable to obtain the variable from the GRIB file")

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
        u_wind10 = extract_from_grb(file_path, 'u_wind10')
        v_wind10 = extract_from_grb(file_path, 'v_wind10')
        for variable_name in variable_names:
            variable = extract_from_grb(file_path, variable_name)
            variables[variable_name] = variable
        
        return Forecast(f'{year}-{month}-{day}', initial_time, lead_time, u_wind10, v_wind10, **variables)
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


def pickle_fold_forecasts(variable_names, i, initial_time, lead_time):
    nobackup = '/net/pc200239/nobackup/users/hakvoort/'

    if i == 1:
        forecasts = fold1_data(variable_names, initial_time, lead_time)
        folder_base = 'fold1data'
    elif i == 2:
        forecasts = fold2_data(variable_names, initial_time, lead_time)
        folder_base = 'fold2data'
    elif i == 3:
        forecasts = fold3_data(variable_names, initial_time, lead_time)
        folder_base = 'fold3data'
    elif i == 0:
        forecasts = validation_data(variable_names, initial_time, lead_time)
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




