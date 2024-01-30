# Prepare Data for dycore:
from pysolar.solar import get_altitude
import datetime
import torch
import numpy as np

class Forcing_Generator():
    def __init__(self,  
                 batch_size:int, 
                 lats:np.array, 
                 lons:np.array, 
                 cst_mask:np.array=None, 
                 forced_masks:np.array=None,
                 device='cpu',
                 ) -> None:
        self._lats = lats
        self._lons = lons
        self.device=device
        self._cst_mask = cst_mask # Earth mask
        self.forced_masks = forced_masks # Sea Temperature, seaice etc
        self.batch_size = batch_size
        self._generate_cst()

    @property
    def cst_mask_dim(self):
        if self.cst_mask is None:
            return 0
        return self.cst_mask.shape[-1]
    
    @property
    def forced_mask_dim(self):
        if self.forced_masks is None:
            print('Warning no forced mask is it intentional?')
            return 0
        return self.forced_masks.shape[-1]

    @property
    def shape(self):
        S = self._lats.shape
        S = (self.batch_size, *S, 3 + 1 + self.cst_mask_dim + self.forced_mask_dim)
        return S
      
    def _generate_cst(self):
        """Generates the constant mask once (lat, lon, frland etc)"""
        self.lats = np.repeat( np.expand_dims( self._lats, axis=0), self.batch_size, axis=0)
        self.lons = np.repeat( np.expand_dims( self._lons, axis=0), self.batch_size, axis=0) 
        lbd_sin_lat = np.vectorize( lambda x : np.cos(2 * np.pi * x / 90))
        lbd_sin_lon = np.vectorize( lambda x : np.cos(2 * np.pi * x / 180))
        lbd_cos_lon = np.vectorize( lambda x : np.sin(2 * np.pi * x / 180))
        lat_sin = lbd_sin_lat(self.lats)
        lon_sin = lbd_sin_lon(self.lons)
        lon_cos = lbd_cos_lon(self.lons)
        self.lat_lon = np.stack([lat_sin, lon_sin, lon_cos], axis=-1)
        if self._cst_mask is None:
            self.cst_mask = None
        else:
            # self._cst_mask is a 
            self.cst_mask = self._cst_mask.to(self.device)
 
    def generate(self, ts, forced_mask=None):
        nl = [self.lat_lon, self.get_toaa(ts)]
        N = np.concatenate(nl, axis=-1)
        X = torch.from_numpy(N).to(self.device)
        nt = [X]
        if not(self.cst_mask is None):
            nt.insert(0, self.cst_mask)
        if not(forced_mask is None):
            self.forced_masks = torch.clone(forced_mask)
            nt.insert(0, self.forced_masks)
        return torch.concatenate( nt, axis=-1).to(torch.float32)

    def format(self, X):
        """From (bs, nb_vars, *) returns (bs, horizontal_points, nb_vars)"""
        X = torch.reshape(X, (self.batch_size, X.size(1), -1) )
        X = torch.swapaxes(X, 1, 2).to(torch.float32)
        return X 
    
    def update(self, ts):
        return self.generate(ts, self.forced_masks)

    def get_latlon(self):
        return

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
        # lats, lons need to be in degrees not radiants
        altitude_deg =  np.stack(    [get_altitude(self.lats[0]*180/np.pi, self.lons[0]*180/np.pi, d) for d in data_dates], axis=0)
        return  np.expand_dims(altitude_deg, axis=-1)
  
    def __len__(self):
        return self.shape[-1]

    @property
    def dimension(self):
        return (self.batch_size, self.__len__(), self.lats)
