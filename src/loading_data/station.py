class Station():
    """
    A station object, which stores the latitude, longitude, altitude of a station.
    We can also set the grid cell.
    """
    def __init__(self, code: str, name: str, latitude: float, longitude: float, altitude: float):
        self.code = code
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.gridcell = None

    def set_grid_cell(self, i: int, j: int) -> None:
        """
        Set the grid cell to a value.
        """
        self.gridcell = (i, j)