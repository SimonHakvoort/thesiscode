import pickle as pkl

class Station():
    """
    A station object, which stores the latitude, longitude, altitude of a station.
    We can also set the grid cell and save this as an attribute.
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

        Arguments:
            i (int)
            j (int)
        """
        self.gridcell = (i, j)


def get_station_info() -> dict:
    """
    Loads station_info.pkl, which contains a dictionary with all the station information

    Arguments:
        None

    Returns:
        dict: dictionary with keys the station numbers and values dictionaries with keys 'lat' and 'lon' and values the latitude and longitude of the station.
    """
    with open('/net/pc200239/nobackup/users/hakvoort/station_info.pkl', 'rb') as f:
        station_info = pkl.load(f)
    return station_info