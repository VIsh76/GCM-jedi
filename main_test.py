from src import Physics, LIN, Normalizer, Dycore, Forcing_Generator, Encode_Decode
from src.forecaster import Forecaster
from src import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import yaml
import tqdm
import torch
import os
import shutil

# %% Initialisation
torch.manual_seed(0)
parameter_path = 'yaml/template.yaml'

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    device = 'cuda'
else:
    device = 'cpu'

with open(parameter_path, 'r') as file:
    parameters = yaml.safe_load(file)

experiment_path = f"{parameters['path']['experiments']}"
graph_path = f"{experiment_path}graph/"
checkpoint_path = f"{experiment_path}checkpoints/"
parameter_file = parameter_path.split('/')[-1]

# Create folders and copy file
os.makedirs(experiment_path, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)
shutil.copyfile(parameter_path, f"{experiment_path}{parameter_file}")

batch_size = parameters['batch_size']
data_path_train = parameters['path']['training']
data_path_test = parameters['path']['testing']
normalizer_path = parameters['path']['normalizers']

pred_surface_vars = parameters['variables']['pred']['surface']
pred_column_vars = parameters['variables']['pred']['column']

cst_surface_vars = parameters['variables']['forced']['cst']
forced_surface_vars = parameters['variables']['forced']['initialized']

#%% Initilized dataloaders:
DL0 = DataLoader(data_path=data_path_train, 
                 batch_size=batch_size, 
                 column_vars=[], 
                 surface_vars=[], 
                 forced_vars=cst_surface_vars, 
                 steps=1, 
                 device='cpu',
                 randomise=False)

(_, _, forced), _, _ = DL0[0]
lats, lons = DL0.get_lat_lon()
FG = Forcing_Generator(batch_size=batch_size, lats=lats, lons=lons, cst_mask=forced, device=device)
del(DL0)

DL_train = DataLoader(data_path=data_path_train, 
                batch_size=batch_size, 
                column_vars=pred_column_vars, 
                surface_vars=pred_surface_vars, 
                forced_vars=forced_surface_vars, 
                steps=1, 
                device=device,
                randomise=True)
DL_test = DataLoader(data_path=data_path_test, 
                batch_size=batch_size, 
                column_vars=pred_column_vars, 
                surface_vars=pred_surface_vars, 
                forced_vars=forced_surface_vars, 
                steps=1, 
                device=device,
                randomise=False)
(col_t1, sur_t1, forced_t1), _, t = DL_train[0]


# %% Architecture :
input_normalizer = {'column':Normalizer(np.load(f'{normalizer_path}input_means_col.npy'), 
                                        np.load(f'{normalizer_path}input_std_col.npy'), 
                                        device), 
                    'surface' :Normalizer(np.load(f'{normalizer_path}input_means_sur.npy'), 
                                          np.load(f'{normalizer_path}input_std_sur.npy'), 
                                          device),
                    }
output_normalizer = {'column':Normalizer(0*np.load(f'{normalizer_path}output_means_col.npy'), 
                                          0.01*np.load(f'{normalizer_path}input_std_col.npy'), 
                                         device), 
                    'surface'  :Normalizer(0.0 * np.load(f'{normalizer_path}output_means_sur.npy'), 
                                           1 + 0 * np.load(f'{normalizer_path}output_std_sur.npy'), 
                                           device)}

from src.model import fill_missing_values
parameters = fill_missing_values(parameters)

model_type = parameters['architecture']['model']
if model_type in ('Physics', 'LIN', 'Encode_Decode'):
    physic_nn = eval(model_type)(parameters=parameters)
else:
    print(f'Model not implemented {model_type}')
    assert(False)

dcore = Dycore()
from src.analysis.architecture_analysis import number_of_parameters
print("Phy number of parameters", number_of_parameters(physic_nn))

forecaster = Forecaster(dcore, physic_nn, input_normalizer, output_normalizer, dt=1)

# %% Initialisation of procedure
from src.training import lbd_scheduler, WeightMSE

## Optimizer
optimizer = torch.optim.AdamW(forecaster.parameters(), lr=1, betas=(0.9,0.95), weight_decay=0.1)
## Scheduler
scheduler_f = lbd_scheduler(**parameters['training']['schedule'])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_f, last_epoch=-1, verbose=False)
## Loss
lat_weights = torch.from_numpy(np.load(f'{normalizer_path}lat_coef.npy')).to(device, dtype=torch.float32)
level_weights = torch.from_numpy(np.load(f'{normalizer_path}lev_coef.npy')).to(device, dtype=torch.float32)

## Region Mask
region_mask = torch.ones_like(sur_t1[:,:,:,[0]]) # t, lat, lon, var (No region mask)
## Weight coefficients:
column_weights = 1 + 0 * input_normalizer['column'].std
column_weights[:,:,:,:,:4] *= 0 # 
surface_weigth = 1 + 0 * input_normalizer['surface'].std

Loss = WeightMSE(surface_weigth=surface_weigth,
                 column_weights=column_weights,
                 level_weights=level_weights,
                 lat_weights=lat_weights,
                 regional_mask=region_mask)
