import numpy as np
import h5py as h5py
from pickle import dump


class GPM():
    """
    Class for GPM data manipulation.
    """
    data, grid, file = None, None, None

    def __init__(self, filename=None, grid=None):
        """
        filename: Full GPM file path (including extension)
        grid    : Python grid like {"max_lat":value, "min_lat":value, "max_lon":value, "min_lon":value}
        """
        try:
            self.file = filename
            self.data = h5py.File(self.file, 'r')
            self.grid = grid
        except Exception:
            raise Exception

    @property
    def minlat(self):
        return np.searchsorted(self.data['Grid/lat'][:], np.min(self.xlat))
    
    @property
    def maxlat(self):
        return np.searchsorted(self.data['Grid/lat'][:], np.max(self.xlat))

    @property
    def minlon(self):
        return np.searchsorted(self.data['Grid/lon'][:], np.min(self.xlon))
    
    @property
    def maxlon(self):
        return np.searchsorted(self.data['Grid/lon'][:], np.max(self.xlon))

    @property
    def xlat(self):
        if self.grid:
            return [lat for lat in self.data['Grid/lat'][:] if self.grid["min_lat"] <= lat <= self.grid["max_lat"]]
        else:
            return self.data['Grid/lat'][:]

    @property
    def xlon(self):
        if self.grid:
            return [lon for lon in self.data['Grid/lon'][:] if self.grid["min_lon"] <= lon <= self.grid["max_lon"]]
        else:
            return self.data['Grid/lon'][:]

    def rain(self, transpose=True):
        if self.grid:
            rain = self.data['Grid/precipitationCal'][0, self.minlon:self.maxlon + 1, self.minlat:self.maxlat + 1]
        else:
            rain = self.data['Grid/precipitationCal'][:]

        if transpose:      
            return np.transpose(rain)
        return rain
        
    def save_as_dict(self, file):
        data = {
            "XLAT":  self.lat,
            "XLONG": self.lon,
            "RAIN":  self.rain,
        }

        with open(file, "wb") as f:
            dump(data, f, protocol=2)
    
    def save_lat_as_txt(self, file):
        np.savetxt(file, self.xlat, delimiter=",", fmt="%7.2f")

    def save_long_as_txt(self, file):
        np.savetxt(file,  self.xlon, delimiter=",", fmt="%7.2f")

    def save_rain_as_txt(self, file):
        np.savetxt(file, self.rain, delimiter=",", fmt="%7.2f")




