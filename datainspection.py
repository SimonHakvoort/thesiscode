
import os
import pygrib

from loadforecasts import fold1_data, fold2_data, fold3_data, make_forecast, validation_data

total = 0

# for year in range(2015, 2020):
#     for month in range(0, 13):
#         for day in range(0, 32):
#             filename_pattern = f'HA40_N25_WINDFCtest_{year}{int(month):02d}{int(day):02d}0000_02400_GB'
#             file_path = os.path.join('/net/pc230023/nobackup/users/ameronge/windwinter/output/', filename_pattern)
#             if os.path.isfile(file_path):
#                 print(f'{year}-{month}-{day}')
#                 total += 1

# print(total)

location = '/net/pc230023/nobackup/users/ameronge/windwinter/output/'
filename_pattern = 'HA40_N25_WINDFCtest_201510010000_02400_GB'
file_path = os.path.join(location, filename_pattern)

grbs = pygrib.open(file_path)

for grb in grbs:
    print(grb)
    print(grb.parameterName)

grbs.close()

