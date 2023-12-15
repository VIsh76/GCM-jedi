# Prepare Data for dycore:
from pysolar.solar import get_altitude
import datetime
import torch
import numpy as np

class Forcing_Generator():
    def __init__(self, mask, lats, lons, batch_size) -> None:
        self._lats = lats
        self._lons = lons
        self._mask = mask
        self.batch_size = batch_size
    
    def _generate_cst(self):
        self.lats = np.repeat( np.expand_dims(self._lats, axis=0), self.batch_size, axis=0), 
        self.lons = np.repeat( np.expand_dims(self._lons, axis=0), self.batch_size, axis=0), 
        self.mask = np.repeat( np.expand_dims(self._mask, axis=0), self.batch_size, axis=0), 
 
    def generate(self, t):
        N = np.concatenate( [self._get_lat_lon(), self._get_toaa(t), self._get_mask()], axis=1)
        X = torch.from_numpy(N)
        X = torch.reshape(X, (X.size(0), X.size(1), -1) ).repeat(self.batch_size, 1, 1)
        X = torch.swapaxes(X, 1, 2).to(torch.float32)
        return X

    def _generate_lat_lon(self):
        lbd_sin_lat = np.vectorize( lambda x : np.cos(2 * np.pi * x / 90))
        lbd_sin_lon = np.vectorize( lambda x : np.cos(2 * np.pi * x / 180))
        lbd_cos_lon = np.vectorize( lambda x : np.sin(2 * np.pi * x / 180))
        lat_sin = np.expand_dims(lbd_sin_lat(self.lats), axis=0)
        lon_sin = np.expand_dims(lbd_sin_lon(self.lons), axis=0)
        lon_cos = np.expand_dims(lbd_cos_lon(self.lons), axis=0)
        return np.concatenate([lat_sin, lon_sin, lon_cos], axis=0)

    def _generate_toaa(self, data_date):
        # Setting the date for tooa function
        data_date = datetime.datetime.strptime(data_date, '%Y-%m-%dT%H:%M:%S')
        data_date = datetime.datetime(year=data_date.year,
                                      month=data_date.month,
                                      day=data_date.day,
                                      hour=data_date.hour,
                                      minute=data_date.minute,
                                      second=data_date.second,
                                      tzinfo=datetime.timezone.utc
                                      )
        altitude_deg = get_altitude(self.lats, self.lons, data_date)
        return np.expand_dims( np.expand_dims(altitude_deg, axis=0), axis=0) #1 var # batch_size
    
    def _get_mask(self):
        return self.mask
  
    def __len__(self):
        return 4