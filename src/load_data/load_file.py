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
                 steps=1, 
                 randomise=False
                 ):
        self.list_of_files = os.listdir(data_path)
        self.data_path = data_path
        self.randomise = randomise
        self.surface_vars = surface_vars
        self.column_vars = column_vars
        self.forced_vars = forced_vars
        self.steps = steps
        self.batch_size = batch_size
        self.ids = np.array([i for i in range(len(self.list_of_files) - steps) ])
        self.initialize_dims()
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.randomise:
            random.shuffle(self.ids)       

    def load_file(self, file_id):
        path = os.path.join(self.data_path + self.list_of_files[file_id])
        warnings.filterwarnings('ignore')
        ds = xr.load_dataset(path)
        ds = ds.drop(['contacts', 'anchor', 'corner_lats', 'corner_lons', 'ncontact'])
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
        if len(self.surface_vars) > 0:
            surface = self.format_surface(X)
            surface = torch.from_numpy(surface)
        if len(self.surface_vars) > 0:
            column = self.format_column(X)
            column  = torch.from_numpy(column)
        if len(self.forced_vars) > 0:
            forced = self.format_forced(X)
            forced = torch.from_numpy(forced)
        # Output shapes are:
        # bs, horizontal_points, nb_var | 
        # bs, horizontal_points, lev, nb_var
        # bs, horizontal_points, nb_var |
        return (column, surface, forced)

    def format_column(self, X):
        column =  X[self.column_vars].to_array().to_numpy() # (var, time, lev, nf, lat, lon)
        assert(column.shape == ((len(self.column_vars), self.batch_size, self.n_levels, self.n_faces, self.n_lats, self.n_lons)))
        # Reshape:
        column = np.reshape(column, (len(self.column_vars), self.batch_size, self.n_levels, -1)  ) # (var, lev, time, horizontal)
        # Reorder axis:
        column = np.moveaxis(column, 2, -1) # lev to last then variable to last
        column = np.moveaxis(column, 0, -1) # variables to last # convolution are handles it latter latter
        return column
    
    def format_surface(self, X):
        surface = X[self.surface_vars].to_array().to_numpy() # (var, time, nf, lat, lon)
        assert(surface.shape == ((len(self.surface_vars), self.batch_size, self.n_faces, self.n_lats, self.n_lons)))
        surface = np.reshape(surface, (len(self.surface_vars), self.batch_size, -1) ) # (var, time, horizontal)
        # Reorder axis:
        surface = np.moveaxis(surface, 0, -1) # variables to last (time, horizontal, var)
        return surface

    def format_forced(self, X):
        forced = X[self.forced_vars].to_array().to_numpy()
        assert(forced.shape == ((len(self.forced_vars), self.batch_size, self.n_faces, self.n_lats, self.n_lons)))
        forced = np.reshape(forced, (len(self.forced_vars), self.batch_size, -1) ) # (var, time, horizontal)
        forced = np.moveaxis(forced, 0, -1) # variables to last
        return forced

    def get_lat_lon(self):
        """ Returns the structure of the lat lon of the variables
            Must be the same shape as 'horizontal' handling of the data
            Here flatten as we reshape the last dimensions with -1 when formating
        """
        ds = self.load_file(0)
        return ds['lats'].values.flatten(), ds['lons'].values.flatten()
 
    def initialize_dims(self):
        ds = self.load_file(0)
        self.n_lons = len(ds['Ydim'])
        self.n_lats  = len(ds['Xdim'])
        self.n_levels = len(ds['lev'])
        self.n_faces  = len(ds['nf'])  

    def dimensions(self):
        out_col = (self.batch_size, self.n_lats*self.n_lons*self.n_faces, self.n_levels, len(self.column_vars))
        out_sur = (self.batch_size, self.n_lats*self.n_lons*self.n_faces, len(self.surface_vars))
        return (out_sur, out_col, out_sur)
    
class ColumnLoader(DataLoader):
    def __init__(self, data_path, surface_vars, column_vars, steps=1, randomise=False, batch_size=2):
        super().__init__(data_path, surface_vars, column_vars, steps, randomise, batch_size)
    
    def dimensions(self):
        out_col = (self.batch_size, self.n_lats,  self.n_faces * self.n_lons, self.n_levels, len(self.column_vars))
        out_sur = (self.batch_size, self.n_lats,  self.n_faces * self.n_lons,  len(self.surface_vars))
        return (out_sur, out_col, out_sur)

    def format_column(self, X):
        column =  X[self.column_vars].to_array().to_numpy() # (var, time, lev, nf, lat, lon)
        assert(column.shape == (len(column), self.batch_size, self.n_levels, self.n_faces, self.n_lats, self.n_lons))
        # Reshape:
        column = np.moveaxis(column, 3, -2) # nf to penultiem
        column = np.reshape(column, (len(self.column_vars), self.batch_size, self.n_levels, self.n_lats, self.n_faces * self.n_lons))
        # Reorder axis:
        column = np.moveaxis(column, 2, -1) # lev to last 
        column = np.moveaxis(column, 0, -1) # variables to last # convolution are handles it latter latter
        return column
    
    def format_surface(self, X):
        surface = X[self.surface_vars].to_array().to_numpy() # (var, time, nf, lat, lon)
        assert(surface.shape == (len(surface), self.batch_size, self.n_faces, self.n_lats, self.n_lons))
        surface = np.moveaxis(surface, 2, -2) # nf to penultiem
        surface = np.reshape(surface, (len(self.surface_vars),  self.batch_size, self.n_lats, self.n_faces * self.n_lons) ) # (var, time, horizontal)
        # Reorder axis:
        surface = np.moveaxis(surface, 0, -1) # variables to last
        return surface

    def format_forced(self, X):
        forced =  X[self.forced_vars].to_array().to_numpy()
        assert(forced.shape == (len(forced), self.batch_size, self.n_faces, self.n_lats, self.n_lons))
        forced = np.moveaxis(forced, 2, -2) # nf to penultiem
        forced = np.reshape(forced, (len(self.forced_vars),  self.batch_size, self.n_lats, self.n_faces * self.n_lons) ) # (var, time, horizontal)
        forced = np.moveaxis(forced, 0, -1) # variables to last
        return forced

    def get_lat_lon(self):
        """ Returns the structure of the lat lon of the variables
            Must be the same shape as 'horizontal' handling of the data
            Here we shift the n_faces as the last dim then considerer they are all longitudes 
            Just like the formatting
        """
        ds = self.load_file(0)
        lats = np.moveaxis(ds['lats'].values, 0, -2)
        lons = np.moveaxis(ds['lons'].values, 0, -2)
        lats = np.reshape(lats, (self.n_lats, self.n_faces*self.n_lons))
        lons = np.reshape(lons, (self.n_lats, self.n_faces*self.n_lons))
        return lats, lons
