from src import Physics, Normalizer, Dycore, Forcing_Generator
from src.forecaster import Forecaster
from src import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import yaml
import tqdm
import warnings
import torch
import os
import shutil

# %% Initialisation
torch.manual_seed(0)
warnings.warn
parameter_path = 'yaml/test_parameters_uvt.yaml'

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    device = 'cuda'
else:
    device = 'cpu'


with open(parameter_path, 'r') as file:
    parameters = yaml.safe_load(file)

experiment_path = f"{parameters['path']['experiments']}"
graph_path = f"{experiment_path}graph"
checkpoint_path = f"{experiment_path}checkpoints/"
parameter_file = parameter_path.split('/')[-1]

# Create folders and copy file
os.makedirs(experiment_path, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)
shutil.copyfile(parameter_path, f"{experiment_path}{parameter_file}")

batch_size = parameters['batch_size']
data_path = parameters['path']['datas']
normalizer_path = parameters['path']['normalizers']

pred_surface_vars = parameters['variables']['pred']['surface']
pred_column_vars = parameters['variables']['pred']['column']

cst_surface_vars = parameters['variables']['forced']['cst']
forced_surface_vars = parameters['variables']['forced']['initialized']

#%% Initilized dataloaders:
DL0 = DataLoader(data_path=data_path, 
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
DL = DataLoader(data_path=data_path, 
                batch_size=batch_size, 
                column_vars=pred_column_vars, 
                surface_vars=pred_surface_vars, 
                forced_vars=forced_surface_vars, 
                steps=1, 
                device=device,
                randomise=True)
(col_t1, surf_t1, forced_t1), (col_t2, surf_t2, _), t  = DL[0]
forced = FG.generate(t, forced_t1)

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
# Switch to 2D embeddings:
parameters['architecture']['column_embedding']['one_d'] = 3
parameters['architecture']['column_embedding']['kernel_size'] = parameters['architecture']['column_embedding']['kernel_size_3d'] 
physic_nn = Physics(parameters=parameters)
dcore = Dycore()

forecaster = Forecaster(dcore, physic_nn, input_normalizer, output_normalizer, dt=1)

# %% Initialisation of procedure
from src.training import lbd_scheduler, WeightMSE

# Optimizer
optimizer = torch.optim.AdamW(forecaster.parameters(), lr=1, betas=(0.9,0.95), weight_decay=0.1)
# Scheduler
scheduler_f = lbd_scheduler(**parameters['training']['schedule'])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_f, last_epoch=-1, verbose=False)
# Loss
lat_weights = torch.from_numpy(np.load(f'{normalizer_path}lat_coef.npy')).to(device, dtype=torch.float32)
level_weights = torch.from_numpy(np.load(f'{normalizer_path}lev_coef.npy')).to(device, dtype=torch.float32)

# Region Mask
region_mask = torch.zeros_like(surf_t1[:,:,:,[0]]) # t, lat, lon, var
p1 = (20, 80)
p2 = (60, 120)
region_mask[0, p1[0]:p2[0], p1[1]:p2[1],:] = 1

column_weights = 1 + 0 * input_normalizer['column'].std
column_weights[:,:,:,:,:4]*=0 # 
surface_weigth = 1 + 0 * input_normalizer['surface'].std

Loss = WeightMSE(surface_weigth=surface_weigth,
                 column_weights=column_weights,
                 level_weights=level_weights,
                 lat_weights=lat_weights,
                 regional_mask=region_mask)

# %% Training
losses_items_avg = []
loss_epoch_avg = []
loss_cst = []

if True:
    n_epochs=2
    step_per_epoch=2
if True:
    n_epochs=5
    step_per_epoch=len(DL)

# %% Test:
checkpoint_path = f"{parameters['path']['experiments']}"
output_folder = f"{checkpoint_path}analysis/"
checkpoint_file = f"{checkpoint_path}checkpointsepoch_4"

os.makedirs(output_folder, exist_ok=True)
forecaster.load_state_dict(torch.load(checkpoint_file))
(col_t1, sur_t1, forced_t1), (col_t2, sur_t2, _), t  = DL[10]
forced = FG.generate(t, forced_t1)

with torch.no_grad():
    col_pred, sur_pred = forecaster.forward(col_t1, sur_t1, forced)

def format_tensor(t:torch.tensor):
    return np.flip( t.to('cpu').numpy()[0], axis=0)

col_pred_n = format_tensor(col_pred) 
sur_pred_n = format_tensor(sur_pred)
col_t2_n  =format_tensor(col_t2) 
sur_t2_n = format_tensor(sur_t2)
col_t1_n  =format_tensor(col_t1) 
sur_t1_n = format_tensor(sur_t1)

var = 't'
id_var = DL.column_vars.index(var)

print(Loss(col_pred, col_t2, sur_pred, sur_pred))
print(Loss(col_t1, col_t2, sur_pred, sur_pred))

plt.imshow(col_pred_n[:,:,-1,id_var]); plt.colorbar(); plt.show();
plt.savefig(f"{output_folder}pred{var}.jpg")
plt.close('all')

plt.imshow(col_t2_n[:,:,-1,id_var]); plt.colorbar(); plt.show();
plt.savefig(f"{output_folder}truth{var}.jpg")
plt.close('all')

data = col_t2_n[:,:,-1,id_var] - col_pred_n[:,:,-1,id_var]
vmax = np.max(abs(data))
plt.imshow(data, cmap='coolwarm', vmax=vmax, vmin=-vmax); plt.colorbar(); plt.show();
plt.close('all')

data = col_t2_n[:,:,-1,id_var] - col_t1_n[:,:,-1,id_var]
vmax = np.max(abs(data))
plt.imshow(data, cmap='coolwarm', vmax=vmax, vmin=-vmax); plt.colorbar(); plt.show();
plt.close('all')


### ALL:
def plot_var_profiles(t1, t2, pred):
    f = plt.figure(figsize=(15, 60))
    titles = ['t1', 't2', 'pred']
    for i, data_0 in enumerate([t1,t2,pred]):
        ax = f.add_subplot(6, 1, i+1)
        ax.set_title(titles[i])
        ax.imshow(data_0)
    
    error_cst = t2 - t1
    error_pred = t2  - pred 
    vmax = max( np.max(abs(error_cst)), np.max(abs(error_pred)))

    # Error of t1 itself
    ax = f.add_subplot(6, 1, 4)
    b = ax.imshow(error_cst, cmap='coolwarm', vmax=vmax, vmin=-vmax);
    ax.set_title('Real change')
    plt.colorbar(b)
    
    # Prediction delta
    ax = f.add_subplot(6, 1, 5)
    data = pred - t1
    vmax = np.max(abs(data))
    b = ax.imshow(data, cmap='coolwarm', vmax=vmax, vmin=-vmax);
    ax.set_title('Predicted delta')
    plt.colorbar(b)
    
    # Comparaison of both error
    delta_error =  abs(error_pred) - abs(error_cst)
    vmax =  np.max(abs(delta_error))
    ax = f.add_subplot(6, 1, 6)
    b = ax.imshow(delta_error  ,cmap='coolwarm' ,vmax=vmax, vmin=-vmax);
    ax.set_title('Comparaison of the Errors (red bad, blue good)')
    plt.colorbar(b)
    
    return f


for var in ['t','u','v']:
    id_var = DL.column_vars.index(var)
    f = plot_var_profiles(  col_t1_n[:,:,-1, id_var],
                        col_t2_n[:,:,-1, id_var],
                        col_pred_n[:,:,-1, id_var],
                  )
    f.savefig(f"{output_folder}{var}.jpg")

var = 'qltot'
id_var = DL.column_vars.index(var)
# Qi, Ql, Qs
f = plot_var_profiles(  np.sum( col_t1_n[:,:, :,id_var] , axis=-1),
                        np.sum( col_t2_n[:,:, :,id_var] , axis=-1),
                        np.sum( col_pred_n[:,:, :,id_var] , axis=-1),
              )
f.savefig(f"{output_folder}{var}.jpg")

var = 'qitot'
id_var = DL.column_vars.index(var)
# Qi, Ql, Qs
f = plot_var_profiles(  np.sum( col_t1_n[:,:, :,id_var] , axis=-1),
                        np.sum( col_t2_n[:,:, :,id_var] , axis=-1),
                        np.sum( col_pred_n[:,:, :,id_var] , axis=-1),                  
              )
f.savefig(f"{output_folder}{var}.jpg")
