import numpy as np


class Forecast:
    def __init__(self, date, initial_time, lead_time, **kwargs):
        self.date = date
        self.initial_time = initial_time
        self.lead_time = lead_time
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.u_component = kwargs.get('U-component of wind m s**-1')
        self.v_component = kwargs.get('V-component of wind m s**-1')
        self.wind_speed = np.sqrt(self.u_component**2 + self.v_component**2)