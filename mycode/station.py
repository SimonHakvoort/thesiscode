

class Station():
    def __init__(self, code, name, latitude, longitude, altitude):
        self.code = code
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.gridcell = None

    def set_grid_cell(self, i, j):
        self.gridcell = (i, j)