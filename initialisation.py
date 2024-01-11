from src.load_data.renorm import compute_renorm_input
from src import DataLoader

import xarray as xr
import numpy as np
import yaml
import os

with open('test_parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

#%% 
data_path = parameters['data_path']
surface_vars = parameters['variables']['pred']['surface']
column_vars = parameters['variables']['pred']['column']

#%% Check if all files have the same 
for i, p in enumerate(os.listdir(data_path)):
    try:
        ds = xr.open_dataset(f"{data_path}{p}")
    except:
        print(f'File {p} \t not read')
    if i ==0:
        ds = xr.open_dataset(f"{data_path}{p}")
        dims = ds.dims
    assert( ds.dims == dims)


#%% Compute means
DL = DataLoader(data_path, 1, surface_vars, column_vars, [])
norm_dict = compute_renorm_input(DL, 10, {}, {})

for data_name in norm_dict:
    np.save(f'data/norms/{data_name}', norm_dict[data_name])
