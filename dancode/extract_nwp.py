import os
import pygrib
import numpy as np
import pickle as pkl
#from tqdm import tqdm

# NWP forecasts folder with raw GRIB files (run for train and test data)
grb_folder  = '/net/pc230023/nobackup/users/ameronge/windwinter/verificationset/'
# Target folder for pickle files at nobackup hard drive
nobackup    = '/net/pc200239/nobackup/users/hakvoort/'
data_folder = nobackup + 'try/'

# Data dimensions
T = 49
grid_size = 300

# Get .grb files, dates and station info
grb_files = sorted([file for file in os.listdir(grb_folder) if (('test' not in file) and ('0000_0' in file))])
grb_dates = sorted(list(set([file.split('_')[3][:8] for file in grb_files])))
with open(nobackup + 'st-info.pkl', 'rb') as file:
    st_info = pkl.load(file)

# For each date store data dict {STATION CODE: SAMPLE}    
#for date in tqdm(grb_dates, position=0, leave=True):
for date in grb_dates:    
    # Init values at -1 to store data
    data = {date + '_' + code : {
        'LOC': info['GRID'],
        'OBS': -np.ones(T),
        'WIND850':-np.ones(T),
        'WIND925':-np.ones(T),
        'ROUGH':-np.ones(T),
        'PRESS':-np.ones(T),
        'KINETIC':-np.ones(T),
        'HUMID':-np.ones(T),
        'GEOPOT':-np.ones(T)}
    for code, info in st_info.items()}
    data['GRID'] = - np.ones((T, grid_size, grid_size))

    # Loop over lead times (.grb files), each .grb file has info about a single lead times for all locations
    grbs = sorted([file for file in grb_files if date in file])
    try:
        for i, grb_file in enumerate(grbs):
            # Load GRIB data (grid format) and calculate wind speeds
            with pygrib.open(grb_folder + grb_file) as grbs:
                wind10  = np.array([grb.values for grb in grbs.select(indicatorOfParameter=[33,34], level=10)])
                wind10  = np.linalg.norm(wind10, axis=0)
                wind850 = np.array([grb.values for grb in grbs.select(indicatorOfParameter=[33,34], level=850)])
                wind850 = np.linalg.norm(wind850, axis=0)
                wind925 = np.array([grb.values for grb in grbs.select(indicatorOfParameter=[33,34], level=925)])
                wind925 = np.linalg.norm(wind925, axis=0)
#               Surface Roughness is not available for all samples, so remove it
#                 rough   = grbs.select(indicatorOfParameter=83,  level=810)[0].values
                press   = grbs.select(indicatorOfParameter=1,   level=0  )[0].values
                kinetic = grbs.select(indicatorOfParameter=200, level=47 )[0].values
                humid   = grbs.select(indicatorOfParameter=52,  level=2  )[0].values
                geopot  = grbs.select(indicatorOfParameter=6,   level=700)[0].values

                # Fetch full grid data for Wind speed at 10m 
                data['GRID'][i] = wind10
                # Fetch local station data for other variables
                for tag, sample in data.items():
                    if not tag == 'GRID':
                        loc = sample['LOC']
                        sample['WIND850'][i] = wind850[loc]
                        sample['WIND925'][i] = wind925[loc]
#                         sample['ROUGH'  ][i] = rough[loc]
                        sample['PRESS'  ][i] = press[loc]
                        sample['KINETIC'][i] = kinetic[loc]
                        sample['HUMID'  ][i] = humid[loc]
                        sample['GEOPOT' ][i] = geopot[loc]
        
        # Dump to pickle file     
        if data:
            with open(data_folder + date + '.pkl', 'wb') as f:
                pkl.dump(data, f)
        else:
            print(f'{date} has no files!')
    except:
        print(date + ' has a problem!')

    print(f"Processed {date} with {len(grbs)} samples")