from src import Physics, LIN, Normalizer, Dycore, Forcing_Generator, Encode_Decode
from src.forecaster import Forecaster
from src import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import yaml
import tqdm
import torch
import os
import sys
import shutil
import datetime
import random
import argparse
    
# %% Initialisation
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# %%  Manual parameters
default_parameter_path = 'yaml/template.yaml'
gpu = True
debug = False

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', default=default_parameter_path)
args = parser.parse_args()

if gpu and torch.cuda.is_available():
    torch.set_default_device('cuda')
    device = 'cuda'
else:
    torch.set_default_device('cpu')
    device = 'cpu'

with open(args.filename, 'r') as file:
    parameters = yaml.safe_load(file)
if not ('experiments_name' in parameters['path']):
    parameters['path']['experiments_name'] = ''

# %% Files:
experiment_path = f"{parameters['path']['experiments']}"
if debug:
    now='debug'
else:
    now = datetime.datetime.strftime(datetime.datetime.now(),  '%Y%m%d_%H%M')
experiment_path = os.path.join(experiment_path, now, ) + '_' + parameters['path']['experiments_name'] +'/'
graph_path = os.path.join(experiment_path, 'graph')+'/'
checkpoint_path = os.path.join(experiment_path, 'checkpoints')+'/'
parameter_file = args.filename.split('/')[-1]

# Create folders and copy file
os.makedirs(experiment_path, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)
os.makedirs(checkpoint_path, exist_ok=True)
shutil.copyfile(args.filename, f"{experiment_path}{parameter_file}")
if not debug:
    sys.stdout = open( os.path.join(experiment_path, f'log.log'), 'w')

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
print(datetime.datetime.now())
print(f"Parameter : \t {args.filename}")
print(f'Output folder : \t {experiment_path}')
print(f"Phy number of parameters : \t {number_of_parameters(physic_nn)}")
print(f"Num datas : \t {len(DL_train)}")
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

# %% Training
loss_items = [] # Loss for all tested items
loss_train = [] # Loss average through an epoch
loss_test = [] # Loss on tests
loss_cst = [] # Prediction = 0
lr_evo = [] # Check learning rate

if 'state_dict' in parameters:
    forecaster.load_state_dict(torch.load(f"{checkpoint_path}{parameters['state_dict']}"))

if True:#Test
    n_epochs=5
    step_per_epoch=1
if True:
    n_epochs=50
    step_per_epoch=len(DL_train)
 
for epoch in range(n_epochs):
    loss_train.append(0)
    loss_test.append(0)
    loss_cst = 0
    for d in tqdm.tqdm(range(step_per_epoch)):
        optimizer.zero_grad()
        (col_t1, surf_t1, forced_t1), (col_t2, surf_t2, _), t  = DL_train[d]
        forced = FG.generate(t, forced_t1)
        ### Reset optim
        ### Forward
        col_pred, surf_pred = forecaster.forward(col_t1, surf_t1, forced)
        l_col, l_sur = Loss(col_pred, col_t2, surf_pred, surf_t2)
        loss = l_col + l_sur
        loss.backward()
        print(t[0], '\t', loss.item())
        torch.nn.utils.clip_grad_norm_(forecaster.parameters(), max_norm=32)
        optimizer.step()
        scheduler.step()
        #Update losses:
        lr_evo.append(scheduler.get_last_lr()[-1])
        assert(lr_evo[-1]>0)
        loss_items.append(loss.item())
        loss_train[-1] += loss.item() / step_per_epoch
        
        ### Cst Loss (baseline)
        l_col0, l_sur0 = Loss(col_t1, col_t2, surf_t1, surf_t2)
        loss0 = l_col0 + l_sur0
        loss_cst += loss0.item() / step_per_epoch

    # On epoch end:
    ## Eval model on test:
    with torch.no_grad():
        for d in tqdm.tqdm(range(len(DL_test))):
            (col_t1, surf_t1, forced_t1), (col_t2, surf_t2, _), t  = DL_test[d]
            forced = FG.generate(t, forced_t1)
            col_pred, surf_pred = forecaster.forward(col_t1, surf_t1, forced)
            l_col, l_sur = Loss(col_pred, col_t2, surf_pred, surf_t2)
            loss = l_col + l_sur
            loss_test[-1] += loss.item() / len(DL_test)
             
    print(f"Epoch {epoch}")
    print(f"Train error \t {loss_train[-1]} \t Test error {loss_test[-1]}   \t Cst error {loss_cst}")    
    DL_train.on_epoch_end()
    DL_test.on_epoch_end()
    # Callbacks early stoppings:
    if epoch<2:
        torch.save(forecaster.state_dict(), f"{checkpoint_path}epoch_{epoch}")        
    if epoch>1: # Stop if increase in error
        # PLOTS:
        plt.plot(loss_items)
        plt.savefig(f'{graph_path}losses_all.jpg')
        plt.show(); plt.close('all')

        plt.plot(loss_train)
        plt.plot(loss_test)
        plt.savefig(f'{graph_path}losses_traintest.jpg')
        plt.show(); plt.close('all')

        plt.plot(np.arange(len(lr_evo)), lr_evo)
        plt.savefig(f'{graph_path}lr.jpg')
        plt.show(); plt.close('all')
        
        if loss_test[-1] > 10*loss_test[-2]:
            print('Warning Exploding gradient')
            del(loss_test[-1])
            del(loss_train[-1])
            del(loss_items[-step_per_epoch:])
            break
        else:# Save the new model
            torch.save(forecaster.state_dict(), f"{checkpoint_path}epoch_{epoch}")

optimizer.zero_grad()


# %% Tests on gradients:
with torch.no_grad():
    forecaster_for_grad = lambda x, y: forecaster(x, y, forced[[0]])
    pert = col_t1[[0]] * 0
    pert[0, 50, 50, :, :] = 1
    pert[0, 50, 0, :, :] = 1
    _, (d_col_ou, d_surf_ou) = torch.autograd.functional.jvp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    _, (d_col_in, d_surf_in) = torch.autograd.functional.vjp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    print('ADJ test :', torch.sum(d_col_ou * pert) -    torch.sum(d_col_in * pert), torch.sum(d_col_in * pert),  torch.sum(d_col_ou * pert))


# %% Test on cube deconstruction
import os

plt.imshow(pert[0,:,:,0,0].to('cpu').numpy()); plt.title('Perturbation')
plt.savefig(f'{graph_path}PT1_perturbation.jpg')
plt.show();plt.close('all')

###############################
plt.imshow(d_col_ou[0,:,:,-1,4].to('cpu').numpy());
plt.colorbar(); plt.title(f"Output_pert, {DL_train.column_vars[4]}")
plt.savefig(f'{graph_path}PT1_tlm_col_output.jpg')
plt.show();plt.close('all')

###############################
plt.imshow(d_col_in[0,:,:,-1,4].to('cpu').numpy());
plt.colorbar(); plt.title(f"Input_pert, {DL_train.column_vars[4]}")
plt.savefig(f'{graph_path}PT1_adj_col_input.jpg')
plt.show();plt.close('all')

plt.imshow(forced[0,:,:,3].to('cpu').numpy());
plt.colorbar(); plt.title(f"Sunlight")
plt.savefig(f'{graph_path}PT1_solar.jpg')
plt.show();plt.close('all')

plt.imshow(surf_pred[0,:,:,-1].to('cpu').detach().numpy())
plt.colorbar(); plt.title(f"Input predicted {DL_train.surface_vars[0]}")
plt.savefig(f'{graph_path}PT_surf_prediction.jpg')
plt.show();plt.close('all')

# Quick ADJ test:
with torch.no_grad():
    forecaster_for_grad = lambda x, y: forecaster(x, y, forced[[0]])
    pert =  torch.randn_like(col_t1[[0]])
    _, (d_col_ou, d_surf_ou) = torch.autograd.functional.jvp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    _, (d_col_in, d_surf_in) = torch.autograd.functional.vjp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    print('ADJ test :', torch.sum(d_col_ou * pert) -    torch.sum(d_col_in * pert), torch.sum(d_col_in * pert),  torch.sum(d_col_ou * pert))

# %% End
print('Hello World')
