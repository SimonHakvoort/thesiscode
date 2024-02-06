#!/usr/bin/env python3
import os
import pygrib
import numpy as np


def extract_variable(year, month, variable_name, initial_time = '0000', forecast = '24'):
    
    # Initialize a dictionary
    dictvariable = {}
    
    # Loop over all dates in the specified year and month
    for day in range(0, 32):
        # Generate the filename pattern for the current date
        filename_pattern = f'HA40_N25_WINDFCtest_{year}{int(month):02d}{int(day):02d}{initial_time}_0{forecast}00_GB'
        
        # Check if the file exists
        file_path = os.path.join('/net/pc230023/nobackup/users/ameronge/windwinter/output/', filename_pattern)
        if os.path.isfile(file_path):
            print(day) 
            # Open the GRIB file
            grbs = pygrib.open(file_path)
            
            # Initialize lists to store u-component and v-component data for the current date
            variable = 0
            
            # Iterate through the messages (fields) in the GRIB file
            for grb in grbs:
                # Check if the current message corresponds to variable_name
                if variable_name in grb.parameterName:
                    # Access the data values for the u-component and store them
                    variable = grb.values

            # Close the GRIB file
            grbs.close()
            
            # Add wind speed to the dictionary with the corresponding date
            dictvariable[f'{year}-{month}-{day}'] = variable
    
    return dictvariable


year = '2017'
month = '11'
wind_speeds = extract_variable(year, month, 'U-component')

for date, wind_speed in wind_speeds.items():
    print(f"Date: {date}, Wind Speed Shape: {wind_speed.shape}")
