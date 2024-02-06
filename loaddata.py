#!/usr/bin/env python3
import os
import pygrib
import numpy as np

def extract_from_grb(path, variable_name):
    grbs = pygrib.open(path)
                
    # Initialize lists to store u-component and v-component data for the current date
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
                
    

def extract_variable(year, month, variable_name, initial_time = '0000', forecast = '24'):
    dictvariable = {}
    if year == '2015' or year == '2016' or year == '2017':
        # Loop over all dates in the specified year and month
        for day in range(0, 32):
            # Generate the filename pattern for the current date
            filename_pattern = f'HA40_N25_WINDFC_{year}{int(month):02d}{int(day):02d}{initial_time}_0{forecast}00_GB'
            
            # Check if the file exists
            file_path = os.path.join('/net/pc230023/nobackup/users/ameronge/windwinter/output/', filename_pattern)
            if os.path.isfile(file_path):
                dictvariable[f'{year}-{int(month):02d}-{int(day):02d}'] = extract_from_grb(file_path, variable_name)

    elif year == '2018' or year == '2019':
        # Loop over all dates in the specified year and month
        for day in range(0, 32):
            # Generate the filename pattern for the current date
            filename_pattern = f'HA40_N25_WindVerificationset_{year}{int(month):02d}{int(day):02d}{initial_time}_0{forecast}00_GB'
            
            # Check if the file exists
            file_path = os.path.join('/net/pc230023/nobackup/users/ameronge/windwinter/verificationset/', filename_pattern)
            if os.path.isfile(file_path):
                dictvariable[f'{year}-{int(month):02d}-{int(day):02d}'] = extract_from_grb(file_path, variable_name)

    return dictvariable            

def extract_wind(year, month, initial_time = '0000', forecast = '24'):
    u_component = extract_variable(year, month, 'U-component of wind m s**-1', initial_time, forecast)
    v_component = extract_variable(year, month, 'V-component of wind m s**-1', initial_time, forecast)

    wind_speeds = {}

    for key in u_component.keys():
        wind_speeds[key] = np.sqrt(u_component[key] ** 2 + v_component[key] ** 2)

    return wind_speeds

def validation_data(variable_name, initial_time = '0000', forecast = '24'):
    data_dict = {}
    for year in range(2018, 2020):
        for month in range(0, 13):
            data_dict.update(extract_variable(year, month, variable_name, initial_time, forecast))

    return data_dict



year = '2017'
month = '11'

wind = extract_wind(year, month)
print(wind["2017-11-12"].shape)


