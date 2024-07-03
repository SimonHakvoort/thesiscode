from typing import Tuple
import numpy as np
import pandas as pd
import os
from src.loading_data.station import Station
import pygrib
import pickle as pkl

def load_stations() -> dict:
    """
    Load station data from CSV files in a specified folder and create a dictionary of Station objects.

    This function reads each CSV file in the designated folder, extracts station information,
    and creates a Station object for each station. The station objects are stored in a dictionary
    with their codes as keys.

    Returns:
        dict: A dictionary where the keys are station codes and the values are Station objects.
    """
    csv_folder = '/net/pc200002/nobackup/users/whan/WOW/KNMI_data/data/'

    station_dict = {}

    for file in os.listdir(csv_folder):
        df = pd.read_csv(csv_folder + file)
        
        if len(df) >= 1:
            code = file.split("_")[2]
            code_in_csv = df['DS_CODE'][0].split("_")[0]
            if code != code_in_csv:
                print("The station code in the file name does not match the station code in the csv file")
                print("The station code in the file name is: ", code)
                print("The station code in the csv file is: ", code_in_csv)
                continue

            name = df['DS_NAME'][0]
            latitude = df['DS_LAT'][0]
            longitude = df['DS_LON'][0]
            altitude = df['DS_ALT'][0]
            station = Station(code, name, latitude, longitude, altitude)
            if code in station_dict:
                if station_dict[code].name != name or station_dict[code].latitude != latitude or station_dict[code].longitude != longitude or station_dict[code].altitude != altitude:
                    print("The station code is already in the dictionary but the name, latitude and longitude do not match")
                    print("The station code is: ", code)
                    print("The station name is: ", name)
                    print("The station latitude is: ", latitude)
                    print("The station longitude is: ", longitude)
                continue
            else:
                station_dict[code] = station

    return station_dict

def find_grid_cell(lat: int, lon: int, lats: np.ndarray, lons: np.ndarray) -> Tuple[int, int]:
    """
    Find the closest grid cell indices for a given latitude and longitude.

    This function takes a latitude and longitude, along with arrays of latitudes and
    longitudes from a grid, and finds the indices of the closest grid cell.

    Args:
        lat (float): The latitude of the point of interest.
        lon (float): The longitude of the point of interest.
        lats (numpy.ndarray): A 2D array of latitudes from the grid.
        lons (numpy.ndarray): A 2D array of longitudes from the grid.

    Returns:
        tuple: A tuple containing the row and column indices (min_lat_index, min_lon_index)
               of the closest grid cell.
    """
    min_lat_diff = 100
    min_lon_diff = 100
    min_lat_index = 0
    min_lon_index = 0
    for i in range(len(lats)):
        lat_diff = abs(lat - lats[i, 0])
        if lat_diff < min_lat_diff:
            min_lat_diff = lat_diff
            min_lat_index = i
    for j in range(len(lons)):
        lon_diff = abs(lon - lons[0, j])
        if lon_diff < min_lon_diff:
            min_lon_diff = lon_diff
            min_lon_index = j
                
    return min_lat_index, min_lon_index

# Load all the stations.
station_info = load_stations()

grb_location = '/net/pc230023/nobackup/users/ameronge/windwinter/output/'
forecast = 'HA40_N25_WINDFC_201603071800_02400_GB'

grbs = pygrib.open(grb_location + forecast)


# Get the latitudes and longitudes of the GRBs
lats, lons = grbs[1].latlons()

# For each station we set the correct grid cell.
for station in station_info.values():
    i, j = find_grid_cell(station.latitude, station.longitude, lats, lons)
    station.set_grid_cell(i, j)

grbs.close()

location = '/net/pc200239/nobackup/users/hakvoort/'

pickle_file_location = location + 'station_info.pkl'

# Load all the information in a pickle file
with open(pickle_file_location, 'wb') as file:
    pkl.dump(station_info, file)

print("The pickle file has been saved to: ", pickle_file_location)


# ## to compare the grid cells i found with the grid cells of daniel
    
# pickle_file_location_daniel = '/net/pc230053/nobackup/users/klein/BackupDaniel/st-info.pkl'

# mypickle = '/net/pc200239/nobackup/users/hakvoort/station_info.pkl'

# with open(mypickle, 'rb') as file:
#     station_dict = pkl.load(file)


# with open(pickle_file_location_daniel, 'rb') as file:
#     data = pkl.load(file)

# for key, value in data.items():
#     if (value['GRID'] != station_dict[key].gridcell):
#         print("The grid cells do not match for station: ", key)
#         print("The grid cell in the pickle file is: ", value['GRID'])
#         print("The grid cell in the station object is: ", station_dict[key].gridcell)

