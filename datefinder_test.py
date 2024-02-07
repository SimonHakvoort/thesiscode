import os
import pandas as pd

path_location = '/net/pc200002/nobackup/users/whan/WOW/KNMI_data/data/'

file1 = 'KIS_TOW_323_W_a_20160101-20161231.csv'
file2 = 'KIS_TOW_229_W_a_20160101-20161231.csv'

s285_1 = 'KIS_TOW_285_W_a_20150101-20151231.csv'
s285_2 = 'KIS_TOW_285_W_a_20160101-20161231.csv'
s285_3 = 'KIS_TOW_285_W_a_20170101-20171231.csv'
s285_4 = 'KIS_TOW_285_W_a_20180101-20181231.csv'
s285_5 = 'KIS_TOW_285_W_a_20190101-20191231.csv'

pd_1 = pd.read_csv(path_location + s285_1)
pd_2 = pd.read_csv(path_location + s285_2)
pd_3 = pd.read_csv(path_location + s285_3)
pd_4 = pd.read_csv(path_location + s285_4)
pd_5 = pd.read_csv(path_location + s285_5)

print(pd_1.shape)
print(pd_2.shape)
print(pd_3.shape)
print(pd_4.shape)
print(pd_5.shape)
