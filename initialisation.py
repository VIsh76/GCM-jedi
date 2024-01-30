from src.load_data.compute_statistic import compute_statistics
from src.load_data.compute_coefficients import compute_lat_coef, compute_level_coef
from src import DataLoader

import xarray as xr
import numpy as np
import yaml
import os

with open('yaml/template.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

#%% 
data_path = parameters['path']['training']
normalizer_path = parameters['path']['normalizers']
output_graph = parameters['path']['graph']+'init/'
surface_vars = parameters['variables']['pred']['surface']
column_vars = parameters['variables']['pred']['column']
os.makedirs(f'{output_graph}', exist_ok=True)

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

#%% Compute coef:
DL = DataLoader(data_path, 1, surface_vars, column_vars, [], steps=1)
(col, sur, _), (col2, sur2, _), t = DL[0] # (time, lat, lon, lev, var)
num_dim = len(sur.shape)

# Lat is in degree:
lats = DL.get_lats()
lats = lats / 180 * np.pi
lat_coef = compute_lat_coef(lats, num_dim, dim=1) #latdim =1
lev_coef = compute_level_coef(np.arange(DL.n_levels), num_dim+1, dim=3) #lev is dim 3 
assert(np.min(lat_coef) > 0)
assert(np.min(lev_coef) > 0)
# %% Animate:

# %% Plot an animation:
print('animating tests')
from src.analysis.animation import update_col, update_force, update_delta_col, generate_animation
import torch

num_steps = 80
vars = []#Set to 0 to avoid animations
vars = ['t','u','v']
for var in vars:
    id_var = DL.column_vars.index(var)
    lev = -1
    ani0 = generate_animation(update_col,
                          kwarg_fct={'id_var':id_var, 'lev':lev, 'DL':DL},
                          kwarg_ani={'frames':np.arange(num_steps)})
    ani0.save(f'{output_graph}{var}_{lev}.gif')
    vmax = abs(col - col2)[0,:,:,lev, id_var].max()
    ani0 = generate_animation(update_delta_col,
                          kwarg_fct={'id_var':id_var, 'lev':lev, 'DL':DL,'vmax':vmax},
                          kwarg_ani={'frames':np.arange(num_steps)},
                          lat=90,
                          lon=180)
    ani0.save(f'{output_graph}delta_{var}_{lev}.gif')

# %% Graph
import matplotlib.pyplot as plt
plt.close('all')
plt.imshow( col[0, :, :, 0, 4].numpy()  ) # 4 is t
plt.savefig(f'{output_graph}T_lev0.jpg')
plt.show(); plt.close()

plt.imshow(col[0, :, :, -1, 4].numpy() ) # 4 is t
plt.savefig(f'{output_graph}T_lev71.jpg')
plt.show(); plt.close()

plt.plot(lat_coef.flatten())
plt.savefig(f'{output_graph}coef_lat.jpg')
plt.show(); plt.close()

plt.plot(lev_coef.flatten())
plt.savefig(f'{output_graph}coef_lev.jpg')
plt.show(); plt.close()

np.save(f'{normalizer_path}lat_coef', lat_coef)
np.save(f'{normalizer_path}lev_coef', lev_coef)


#%% Compute means
DL = DataLoader(data_path, 1, surface_vars, column_vars, [], steps=1)
norm_dict = compute_statistics(DL, 80)

os.makedirs(normalizer_path, exist_ok=True)
for data_name in norm_dict:
    np.save(f'{normalizer_path}{data_name}', norm_dict[data_name])
    print('Saving \t', f"{normalizer_path}{data_name}")
print('Over')

