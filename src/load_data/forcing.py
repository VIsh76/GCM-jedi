# Prepare Data for dycore:
from pysolar.solar import get_altitude
import datetime
import torch
import numpy as np

class Forcing_Generator():
    def __init__(self, lats, lons, batch_size) -> None:
        self._lats = lats
        self._lons = lons
        #self._mask = mask
        self.batch_size = batch_size
        self._generate_cst()
  
    def _generate_cst(self):
        self.lats = np.repeat( np.expand_dims(self._lats, axis=0), self.batch_size, axis=0)
        self.lons = np.repeat( np.expand_dims(self._lons, axis=0), self.batch_size, axis=0) 
        lbd_sin_lat = np.vectorize( lambda x : np.cos(2 * np.pi * x / 90))
        lbd_sin_lon = np.vectorize( lambda x : np.cos(2 * np.pi * x / 180))
        lbd_cos_lon = np.vectorize( lambda x : np.sin(2 * np.pi * x / 180))
        self.lat_sin = lbd_sin_lat(self.lats)
        self.lon_sin = lbd_sin_lon(self.lons)
        self.lon_cos = lbd_cos_lon(self.lons)
        #self.mask = np.repeat( np.expand_dims(self._mask, axis=0), self.batch_size, axis=0), 
 
    def generate(self, ts):
        N = np.concatenate( [self.get_latlon(), self.get_toaa(ts)], axis=1)
        X = torch.from_numpy(N)
        X = torch.reshape(X, (self.batch_size, X.size(1), -1) )
        X = torch.swapaxes(X, 1, 2).to(torch.float32)
        return X


    def get_latlon(self):
        return np.stack([self.lat_sin, self.lon_sin, self.lon_cos], axis=1)


    def get_toaa(self, data_dates):
        # Setting the date for tooa function
        data_dates = [datetime.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S') for d in data_dates]
        data_dates = [datetime.datetime(year=data_date.year,
                                      month=data_date.month,
                                      day=data_date.day,
                                      hour=data_date.hour,
                                      minute=data_date.minute,
                                      second=data_date.second,
                                      tzinfo=datetime.timezone.utc
                                      ) for data_date in data_dates]
        altitude_deg =  np.stack(    [get_altitude(self.lats[0], self.lons[0], d) for d in data_dates], axis=0)
        return  np.expand_dims(altitude_deg, axis=1)
    
    def _generate_mask(self):
        return self.mask
  
    def __len__(self):
        return 4