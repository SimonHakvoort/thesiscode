import os
import pickle as pkl

# retrieve the forecasts from the pickle files and returns them as a list
def get_folds():
    fold0 = get_fold_i(0)
    fold1 = get_fold_i(1)
    fold2 = get_fold_i(2)
    fold3 = get_fold_i(3)

    return fold0, fold1, fold2, fold3

def get_fold_i(i):
    foldi = []

    for file in os.listdir(f'/net/pc200239/nobackup/users/hakvoort/fold{i}data/'):
        if file.endswith('.pkl'):
            with open(f'/net/pc200239/nobackup/users/hakvoort/fold{i}data/' + file, 'rb') as f:
                forecast = pkl.load(f)
                foldi.append(forecast)
    
    return foldi

def get_station_info():
    with open('/net/pc200239/nobackup/users/hakvoort/station_info.pkl', 'rb') as f:
        station_info = pkl.load(f)
    return station_info