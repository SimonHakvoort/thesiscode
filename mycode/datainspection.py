import pandas as pd


csv_folder = '/net/pc200002/nobackup/users/whan/WOW/KNMI_data/data/'

file = 'KIS_TOW_280_W_23t_20150101-20151231.csv'

df = pd.read_csv(csv_folder + file)

df = df.rename(columns={'IT_DATETIME':'DATETIME', 'FF_10M_10':'OBS'}, inplace=False)
df = df[['DATETIME', 'OBS']]

time = '20151221_000000_000000'
time2 = '20151231_234000_000000'

for i in range(49500, 49601):
    print(df.loc[i])