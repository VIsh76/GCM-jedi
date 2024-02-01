# %%
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
parameter_path = 'yaml/full.yaml'
cpu=False

if cpu:
    torch.set_default_device('cpu')
    device = 'cpu'
else:
    torch.set_default_device('cuda')
    device = 'cuda'
    

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

(_, _, _forced), _, _ = DL0[0]
lats, lons = DL0.get_lat_lon()
FG = Forcing_Generator(batch_size=batch_size, lats=lats, lons=lons, cst_mask=_forced, device=device)
del(DL0, _forced)

DL_train = DataLoader(data_path=data_path_train, 
                batch_size=batch_size, 
                column_vars=pred_column_vars, 
                surface_vars=pred_surface_vars, 
                forced_vars=forced_surface_vars, 
                steps=1, 
                device=device,
                randomise=False)
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

# %% [markdown]
# ### 

# %% [markdown]
# ## Load State dict

# %%
state_dict_file = f'{checkpoint_path}epoch_29'
state_name = 'e29'
forecaster.load_state_dict(torch.load(state_dict_file))

# %%
with torch.no_grad():
    (col_t1, surf_t1, _forced), (col_t2, surf_t2, forced_t2), t1  = DL_train[0]
    forced_t1 = FG.generate(t1, _forced)
    forecaster_for_grad = lambda x, y: forecaster(x, y, forced_t1[[0]])
    pert = col_t1[[0]] * 0
    pert[0, 10, 10, :, :] = 1
    _, (d_col_ou, d_surf_ou) = torch.autograd.functional.jvp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    _, (d_col_in, d_surf_in) = torch.autograd.functional.vjp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    print('ADJ test :', torch.sum(d_col_ou * pert) -    torch.sum(d_col_in * pert), torch.sum(d_col_in * pert),  torch.sum(d_col_ou * pert))

# %%
col_t1.device, surf_t1.device, forced_t1.device

# %%
with torch.no_grad():
    forecaster_for_grad = lambda x, y: forecaster(x, y, forced_t1[[0]])
    pert =  torch.randn_like(col_t1[[0]])
    _, (d_col_ou, d_surf_ou) = torch.autograd.functional.jvp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    _, (d_col_in, d_surf_in) = torch.autograd.functional.vjp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    print('ADJ test :', torch.sum(d_col_ou * pert) -    torch.sum(d_col_in * pert), torch.sum(d_col_in * pert),  torch.sum(d_col_ou * pert))


# %% [markdown]
# # Animate Data

# %%
from src.analysis.architecture_analysis import get_all_layers_name, get_activation

forecaster_layer_names = get_all_layers_name(forecaster)
forecaster_layer_names = ['forecaster.'+i for i in forecaster_layer_names]

def generate_hooks(model, layers:dict()):
    """
    Model: is the model (forecaster)
    layer_name: dict with names as key and layers object as values
    """
    activation = dict()
    for layer_name in layers:
        eval(layer_name).register_forward_hook(get_activation(layer_name, activation))
    return activation

activation = generate_hooks(forecaster, forecaster_layer_names[:9])

with torch.no_grad():
    (col_t1, surf_t1, forced_t1), (col_t2, surf_t2, forced_t2), t1  = DL_train[0]
    forced_t1 = FG.generate(t1, forced_t1)
    output1 = forecaster(col_t1, surf_t1, forced_t1)

# %% [markdown]
# # Layers

# %%
for name in activation:
    print(name,': \t ', activation[name].shape)

# %%
def plot_weights(data, n_cols, subfigsize, titles=[]):
    """Return a figure for each last dim of the data, returns 
    a plot of it data is shape (lat, lon, D)
    Args:
        data (np.array): (lat, lon, D)
        n_cols (int): num of cols
        subfigsize(int, int): size of smaller figures
    """
    n_plots = data.shape[-1]
    n_lignes = (n_plots -1)// n_cols + 1
    f = plt.figure( figsize=(subfigsize[1]*n_cols, subfigsize[0]*n_lignes))
    for d in range(data.shape[-1]):
        ax = f.add_subplot(n_lignes, n_cols, d+1)
        ax.imshow(data[:,:, d])
        if len(titles)>d:
            ax.set_title(titles[d])
    return f

data =  activation['forecaster.physic.col_embedding.col_embd_block_4.conv'].to('cpu').numpy()[0]
data = np.swapaxes(data, 0, -1)[0, :, :]
f = plot_weights(data, 3, (5, 5) );
f.savefig( f'{graph_path}/column_embd_lev0.jpg')

data =  activation['forecaster.physic.col_embedding.col_embd_block_4.conv'].to('cpu').numpy()[0]
data = np.swapaxes(data, 0, -1)[0, :, :]
f = plot_weights(data, 3, (5, 5) )
f.savefig( f'{graph_path}/column_embd_lev-1.jpg');

# %%
for i in range(3):
    f = plot_weights(  activation[f'forecaster.physic.sur_embedding.sur_embd_block_{i}.conv'].to('cpu').numpy()[0], 3, (5, 5) )
    f.savefig( f'{graph_path}/{state_name}_surface_embd_{i}.jpg' )

# %% [markdown]
# # Prediction

# %%
from src.analysis.graph import plot_var_profiles

def plot_var_profiles(t1, t2, pred):
    """
    t1, t2, pred size [x,y]
    """
    f = plt.figure(figsize=(10, 35))
    titles = ['t1', 't2', 'pred']
    for i, data_0 in enumerate([t1,t2,pred]):
        ax = f.add_subplot(7, 1, i+1)
        ax.set_title(titles[i])
        ax.imshow(data_0)
    
    data1 = t1 - t2
    data2 = pred - t2
    vmax = max(np.max(data1), np.max(data2))
    
    ax = f.add_subplot(7, 1, 4)
    ln = ax.imshow(data1, cmap='coolwarm', vmax=vmax, vmin=-vmax);
    ax.set_title('Static error values')
    plt.colorbar(ln)
    
    ax = f.add_subplot(7, 1, 5)
    ln = ax.imshow(data2, cmap='coolwarm', vmax=vmax, vmin=-vmax);
    ax.set_title('Prediction Error values')
    plt.colorbar(ln)

    ###########################################################################
    ax = f.add_subplot(7, 1, 6)
    ln = ax.imshow(abs(data1), vmax=vmax)
    ax.set_title('Static Error abs values')
    plt.colorbar(ln)
    
    ax = f.add_subplot(7, 1, 7)
    ln = ax.imshow(abs(data2), vmax=vmax);
    ax.set_title('Prediction Error abs values')
    plt.colorbar(ln)

    return f
for var in ['u','v','t']:
    id_var = DL_train.column_vars.index(var)
    id_lev = -1
    col_t1n = col_t1.to('cpu').numpy()[0,:,:, id_lev, id_var]
    col_t2n = col_t2.to('cpu').numpy()[0,:,:, id_lev, id_var]
    output_1n = output1[0].to('cpu').numpy()[0,:,:, id_lev, id_var]
    f = plot_var_profiles(col_t1n, col_t2n, output_1n)
    f.savefig(f"{graph_path}{state_name}_output_result{var}{id_lev}.png")

# %% [markdown]
# ### 

# %%
col_t1.size()

# %%
with torch.no_grad():
    (col_t1, surf_t1, forced_t1), (col_t2, surf_t2, forced_t2), t1  = DL_train[0]
    forced_t1 = FG.generate(t1, forced_t1)
    output1 = forecaster(col_t1, surf_t1, forced_t1)

# %%



