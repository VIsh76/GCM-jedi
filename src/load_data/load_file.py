import numpy as np 
import os
import random
import torch
import xarray as xr


class DataLoader():
    def __init__(self, 
                 data_path, 
                 surface_vars,
                 column_vars,
                 steps=1, 
                 randomise=False):
        self.list_of_files = os.listdir(data_path)
        self.data_path = data_path
        self.randomise = randomise
        self.surface_vars = surface_vars
        self.column_vars = column_vars
        self.steps = steps
        self.batch = 1
        self.ids = [i for i in range(len(self.list_of_files) - steps) ]
        self.initialize_dims()
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.randomise:
            random.shuffle(self.ids)       

    def load_file(self, file_id):
        path = os.path.join(self.data_path + self.list_of_files[file_id])
        ds = xr.load_dataset(path)
        ds = ds.drop(['contacts', 'anchor', 'corner_lats', 'corner_lons'])
        return ds
        
    def __getitem__(self, element):
        file_x_id = self.ids[element]
        file_y_id = file_x_id + self.steps
        X = self.load_file(file_x_id)
        t = str(X['time'].values[0]).split('.')[0]
        X = self.format(X)
        Y = self.load_file(file_y_id)
        Y = self.format(Y)
        return X, Y, t

    def format(self, X:xr.DataArray):
        surface = np.squeeze(X[self.surface_vars].to_array().to_numpy())
        column =  np.squeeze(X[self.column_vars].to_array().to_numpy())
        surface = np.reshape(surface, (self.batch, len(self.surface_vars), -1))
        column = np.reshape(column, (self.batch,  len(self.column_vars), self.n_levels, -1))
        
        surface = torch.from_numpy(surface)
        column  = torch.from_numpy(column)
        # Place variables as the last values
        surface = torch.swapaxes(surface, 1, 2).to(torch.float32)
        column  = torch.swapaxes(column, 1, 3).to(torch.float32) # Might switch
        return (surface, column)

    def get_lat_lon(self):
        ds = self.load_file(0)
        return ds['lats'].values, ds['lons'].values
 
    def initialize_dims(self):
        ds = self.load_file(0)
        self.n_lons = len(ds['lons'])
        self.n_lats     = len(ds['lats'])
        self.n_levels   = len(ds['lev'])
        self.n_faces    = len(ds['nf'])  