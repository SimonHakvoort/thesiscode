#!/usr/bin/env python3
import os
from typing import Tuple
import pygrib
from src.loading_data.forecast import Forecast
import pickle as pkl


def ObtainParameterName(parameter_name: str) -> Tuple[int, int]:
    """
    For a given variable it returns a tuple, where the first element is the indicatorOfParameter for the grb and the second element is the level.

    Arguments:
        parameter_name (str): name of the weather variable

    Returns:
        Tuple[int, int] with the location in the grb file.
    """ 
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


def extract_from_grb(path: str, variable_name: str):
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

def test_data(variable_names: list[str], initial_time: str = '0000', lead_time: str = '48') -> list[Forecast]:
    """
    Makes Forecast objects for the test data.

    Arguments:
        variable_names (list[str]): a list of variables to include in the Forecast objects.
        initial_time (str, 'HHMM'): the time when the forecast is initialized.
        lead_time (str, 'HH'): the lead time of the forecast.

    Returns:
        A list of Forecast objects for the test data.
    """
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

def fold1_data(variable_names: list[str], initial_time: str = '0000', lead_time: str = '48') -> list[Forecast]:
    """
    Makes Forecast objects for fold 1.

    Arguments:
        variable_names (list[str]): a list of variables to include in the Forecast objects.
        initial_time (str, 'HHMM'): the time when the forecast is initialized.
        lead_time (str, 'HH'): the lead time of the forecast.

    Returns:
        A list of Forecast objects for fold 1.
    """
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

def fold2_data(variable_names: list[str], initial_time: str = '0000', lead_time: str = '48') -> list[Forecast]:
    """
    Makes Forecast objects for fold 2.

    Arguments:
        variable_names (list[str]): a list of variables to include in the Forecast objects.
        initial_time (str, 'HHMM'): the time when the forecast is initialized.
        lead_time (str, 'HH'): the lead time of the forecast.

    Returns:
        A list of Forecast objects for fold 2.
    """
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

def fold3_data(variable_names: list[str], initial_time: str = '0000', lead_time: str = '48') -> list[Forecast]:
    """
    Makes Forecast objects for fold 3.

    Arguments:
        variable_names (list[str]): a list of variables to include in the Forecast objects.
        initial_time (str, 'HHMM'): the time when the forecast is initialized.
        lead_time (str, 'HH'): the lead time of the forecast.

    Returns:
        A list of Forecast objects for fold 3.
    """
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


def pickle_fold_forecasts(variable_names: list, i: int, initial_time: str, lead_time: str) -> None:
    """
    Pickles a the Forecast objects, such that they can be loaded in later. 
    No observations are added. We denote with fold 0 the test data. 
    Saves the Forecast under the name 'date_leadtime.pkl'.

    Arguments:
        variable_names (list): a list of the variables that need to be included.
        i (int): fold number, 0, 1, 2 or 3.
        initial_time (str): the initial time, in the form 'HHMM'
        lead_time (str): the lead time, in the form 'HH'
    """
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
        forecasts = test_data(variable_names, initial_time, lead_time)
        folder_base = 'fold0data'
    else:
        raise ValueError("Invalid value for i. Expected values: 0 (test data), 1, 2, or 3.")

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




