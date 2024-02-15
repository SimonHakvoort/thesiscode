
import pickle as pkl
from loadforecasts import fold1_data, pickle_fold_forecasts
import pickle as pkl
from addobservations import addallobservations



# variables = ['U-component of wind m s**-1', 'V-component of wind m s**-1', 'R Relative humidity %']
# initial_time = '0000'
# lead_time = '36'
# for i in range(0, 2):
#     pickle_fold_forecasts(variables, i, initial_time, lead_time)
#     print("done with fold: ", i)
#     addallobservations('fold' + str(i) + 'data/')
#     print("done with adding observations to fold: ", i)


variables = ['press', 'kinetic', 'humid', 'geopot']
initial_time = '0000'
lead_time = '36'

for i in range(0, 4):
    pickle_fold_forecasts(variables, i, initial_time, lead_time)
    print("done with fold: ", i)
    addallobservations('fold' + str(i) + 'data/')
    print("done with adding observations to fold: ", i)

# #print informatoin about the first forecast

# with open('/net/pc200239/nobackup/users/hakvoort/fold1data/2015-10-01_' + lead_time + '.0.pkl', 'rb') as file:
#     forecast = pkl.load(file)
#     print("The forecast date is: ", forecast.date)
#     print("The initial time is: ", forecast.initial_time)
#     print("The lead time is: ", forecast.lead_time)
#     #print("The predicted wind speed is: ", forecast.wind_speed)
#     #print("The variables we use are: ")
#     #for variable in variables:
#     #    print(SimplifyName(variable), " with dimensions: ", getattr(forecast, SimplifyName(variable)).shape)
#     print("The observations are: ")
#     for key, value in forecast.observations.items():
#         print("Station code: ", key, " Observation: ", value[0], " Date and time: ", value[1])