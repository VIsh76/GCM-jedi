import numpy as np 
import os
import random
import torch
import xarray as xr
import warnings

class DataLoader():
    def __init__(self,
                 data_path,
                 batch_size,
                 surface_vars,
                 column_vars,
                 forced_vars,
                 steps, 
                 randomise=False,
                 device='cpu',
                 dt=''
                 ):
        assert(steps > 0)
        self.list_of_files = os.listdir(data_path)
        self.list_of_files = [l for l in self.list_of_files if l.split('.')[-1]=='nc4' ]
        self.list_of_files.sort()
        self.data_path = data_path
        self.randomise = randomise
        self.surface_vars = surface_vars
        self.column_vars = column_vars
        self.forced_vars = forced_vars
        self.steps = steps
        self.batch_size = batch_size
        self.device = device
        self.ids = np.array([i for i in range(len(self.list_of_files) - steps) ])
        self.initialize_dims()
        self.on_epoch_end()

    def __len__(self):
        return (len(self.list_of_files) - self.steps) // self.batch_size

    def on_epoch_end(self):
        if self.randomise:
            random.shuffle(self.ids)       

    def load_file(self, file_id):
        path = os.path.join(self.data_path + self.list_of_files[file_id])
        warnings.filterwarnings('ignore')
        ds = xr.load_dataset(path) # {'lon': 180, 'lat': 91, 'lev': 72, 'time': 1}
        return ds
        
    def __getitem__(self, element):
        file_x_ids = self.ids[element*self.batch_size:(element+1)*self.batch_size]
        file_y_ids = file_x_ids + self.steps
        # TD concat in term of array or concat in numpy?
        # TD Load files in parallel
        t = []
        X = []
        Y = []
        for file_id in file_x_ids:
            X.append(self.load_file(file_id))
        for file_id in file_y_ids:
            Y.append(self.load_file(file_id))
        X = xr.concat(X, dim='time')
        Y = xr.concat(Y, dim='time')
        t = [  str(T).split('.')[0] for T in (X['time'].values)]
        X = self.format(X)
        Y = self.format(Y)
        return X, Y, t

    def format(self, X:xr.DataArray):
        surface = []
        column = []
        forced = []
        if len(self.column_vars) > 0:
            column = self.format_column(X)
            column = torch.from_numpy(column).to(self.device)
        if len(self.surface_vars) > 0:
            surface = self.format_surface(X)
            surface = torch.from_numpy(surface).to(self.device)
        if len(self.forced_vars) > 0:
            forced = self.format_forced(X)
            forced = torch.from_numpy(forced).to(self.device)
        # Output shapes are:
        # bs, horizontal_points, nb_var | 
        # bs, horizontal_points, lev, nb_var
        # bs, horizontal_points, nb_var |
        return (column, surface, forced)

    def format_column(self, X):
        column =  X[self.column_vars].to_array().to_numpy() # (var, time, lev, lat, lon)
        assert(column.shape == ((len(self.column_vars), self.batch_size, self.n_levels, self.n_lats, self.n_lons)))
        # Reorder axis:
        column = np.moveaxis(column, 2, -1) # lev to last then variable to last
        column = np.moveaxis(column, 0, -1) # variables to last # convolution are handles it latter latter
        return column
    
    def format_surface(self, X):
        surface = X[self.surface_vars].to_array().to_numpy() # (var, time, lat, lon)
        assert(surface.shape == ((len(self.surface_vars), self.batch_size, self.n_lats, self.n_lons)))
        # Reorder axis:
        surface = np.moveaxis(surface, 0, -1) # variables to last (time, horizontal, var)
        return surface

    def format_forced(self, X):
        forced = X[self.forced_vars].to_array().to_numpy()
        assert(forced.shape == ((len(self.forced_vars), self.batch_size, self.n_lats, self.n_lons)))
        forced = np.moveaxis(forced, 0, -1) # variables to last
        return forced

    def get_lat_lon(self):
        """ Returns the structure of the lat lon of the variables
            Must be the same shape as 'horizontal' handling of the data
            Here flatten as we reshape the last dimensions with -1 when formating
        """
        ds = self.load_file(0)
        lats = np.repeat(np.expand_dims(ds['lat'].values.flatten(), axis=1), repeats=self.n_lons, axis=1) / 360 * np.pi * 2
        lons = np.repeat(np.expand_dims(ds['lon'].values.flatten(), axis=0), repeats=self.n_lats, axis=0) / 360 * np.pi * 2
        return lats, lons
    
    def get_lats(self):
        ds = self.load_file(0)
        return ds['lat'].values
        
 
    def initialize_dims(self):
        ds = self.load_file(0)
        self.n_lons = len(ds['lon'])
        self.n_lats  = len(ds['lat'])
        self.n_levels = len(ds['lev'])

    def dimensions(self):
        out_col = (self.batch_size, self.n_lats, self.n_lons, self.n_levels, len(self.column_vars))
        out_sur = (self.batch_size, self.n_lats, self.n_lons, len(self.surface_vars))
        out_frc = (self.batch_size, self.n_lats, self.n_lons, len(self.forced_vars))
        return {'surface':out_sur, 'column':out_col, 'forced':out_frc}
