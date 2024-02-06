
import os

total = 0

for year in range(2015, 2020):
    for month in range(0, 13):
        for day in range(0, 32):
            filename_pattern = f'HA40_N25_WINDFCtest_{year}{int(month):02d}{int(day):02d}0000_02400_GB'
            file_path = os.path.join('/net/pc230023/nobackup/users/ameronge/windwinter/output/', filename_pattern)
            if os.path.isfile(file_path):
                print(f'{year}-{month}-{day}')
                total += 1

print(total)