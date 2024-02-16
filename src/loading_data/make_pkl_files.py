
from loadforecasts import pickle_fold_forecasts
from addobservations import addallobservations


variables = ['press', 'kinetic', 'humid', 'geopot']
initial_time = '0000'
lead_time = '48'

for i in range(0, 4):
    pickle_fold_forecasts(variables, i, initial_time, lead_time)
    print("done with fold: ", i)
    addallobservations('fold' + str(i) + 'data/')
    print("done with adding observations to fold: ", i)

