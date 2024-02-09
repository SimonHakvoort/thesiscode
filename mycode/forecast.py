import numpy as np
from datetime import datetime, timedelta


class Forecast:
    def __init__(self, date, initial_time, lead_time, **kwargs):
        year, month, day = map(int, date.split('-'))
        hour, minute = divmod(int(initial_time), 100)
        self.date = datetime(year, month, day)
        self.initial_time = datetime(year, month, day, hour, minute)
        self.lead_time = timedelta(hours=int(lead_time))
        for key, value in kwargs.items():
            attribute_name = SimplifyName(key)
            setattr(self, attribute_name, value)
        self.wind_speed = np.sqrt(self.U_componentofwindms_1**2 + self.V_componentofwindms_1**2)

        self.observations = {}

    def add_observation(self, station_code, observation, date_time):
        # add the observation and date_time as tuple to the dictionary
        self.observations[station_code] = (observation, date_time)

def SimplifyName(string):
    # Remove spaces and special characters from the attribute name
    attribute_name = ''.join(c if c.isalnum() else '_' for c in string.replace(' ', ''))
    while '__' in attribute_name:
        attribute_name = attribute_name.replace('__', '_')
    
    return attribute_name