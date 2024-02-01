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
#%% Parameters
max_steps = 1


#%% Initilized dataloaders:
DL0 = DataLoader(data_path=data_path_train, 
                 batch_size=batch_size, 
                 column_vars=[], 
                 surface_vars=[], 
                 forced_vars=cst_surface_vars, 
                 steps=1, 
                 device='cpu',
                 randomise=False)

L, _ = DL0[0]
(_, _, _forced) = L[0]
lats, lons = DL0.get_lat_lon()
FG = Forcing_Generator(batch_size=batch_size, lats=lats, lons=lons, cst_mask=_forced, device=device)
del(DL0)

DL_train = DataLoader(data_path=data_path_train, 
                batch_size=batch_size, 
                column_vars=pred_column_vars, 
                surface_vars=pred_surface_vars, 
                forced_vars=forced_surface_vars, 
                steps=max_steps, 
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

L, t_list = DL_train[0]
(col_t1, sur_t1, _forced) = L[0]
forced_t1 = FG.generate(t_list[0], _forced)


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
optimizer = torch.optim.AdamW(forecaster.parameters(), lr=1, betas=(0.9, 0.95), weight_decay=0.1)
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

if True:
    n_epochs=2
    step_per_epoch=2
if True:
    n_epochs=50
    step_per_epoch=len(DL_train)

time_coef = [1, 1, 0.2] # Coefficient of the loss based on the step [0 is not used]
for epoch in range(n_epochs):
    print(f'Epoch {epoch}')
    loss_train.append(0)
    loss_cst.append(0)
    loss_test.append(0)
    for d in range(step_per_epoch):
        optimizer.zero_grad()
        data_list, t_list  = DL_train[d, max_steps]
#        ((col_t1, surf_t1, _forced), (col_t2, surf_t2, _)), t_list  = DL_train[d, max_steps]
        col_pred, sur_pred, _forced = data_list[0]
        col_t0, sur_t0, _ = data_list[0]
        loss = 0
        loss0 = 0
        for i in range(1, len(data_list)):
            # predictive model:
            force_in = FG.generate(t_list[i], _forced)
            col_truth, sur_truth, _ = data_list[i]
            col_pred, sur_pred = forecaster.forward(col_pred, sur_pred, force_in)
            l_col, l_sur = Loss(col_pred, col_truth, sur_pred, sur_truth)
            loss += (l_col + l_sur) * time_coef[i]
            print(loss)
            # cst model
            l_col0, l_sur0 = Loss(col_t0, col_truth, sur_t0, sur_truth)
            loss0 += (l_col0 + l_sur0) * time_coef[i]

        ### Reset optim, update:
        loss.backward()
        loss_items.append(loss.item())
        torch.nn.utils.clip_grad_norm_(forecaster.parameters(), max_norm=32)
        optimizer.step()
        scheduler.step()

        #Update losses:
        lr_evo.append(scheduler.get_last_lr()[-1])
        loss_cst[-1] += loss0.item() / step_per_epoch
        loss_train[-1] += loss.item() / step_per_epoch

    # On epoch end:
    ## Eval model on test:
    with torch.no_grad():
        for d in tqdm.tqdm(range(len(DL_test))):
            ((col_t1, surf_t1, _forced), (col_t2, surf_t2, _)), t_list  = DL_test[d, 1]
            forced_t1 = FG.generate(t_list[0], _forced)
            col_pred, surf_pred = forecaster.forward(col_t1, surf_t1, forced_t1)
            l_col, l_sur = Loss(col_pred, col_t2, surf_pred, surf_t2)
            loss = l_col + l_sur
            loss_test[-1] += loss.item() / len(DL_test)
             
    print(f"Epoch {epoch}")
    print(f"Train error \t {loss_train[-1]} \t Test error {loss_test[-1]}   \t Cst error {loss_cst[-1]}")    
    DL_train.on_epoch_end()
    DL_test.on_epoch_end()
    # Callbacks early stoppings:
    if epoch>=2: # Stop if increase in error
        if loss_test[-1] > 10*loss_test[-2]:
            print('Warning Exploding gradient')
            del(loss_test[-1])
            del(loss_train[-1])
            del(loss_items[-step_per_epoch:])
            break
        else:# Save the new model
            torch.save(forecaster.state_dict(), f"{checkpoint_path}epoch_{epoch}")

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

# %% Tests on gradients:
with torch.no_grad():
    forecaster_for_grad = lambda x, y: forecaster(x, y, forced_t1[[0]])
    pert = col_t1[[0]] * 0
    pert[0, 10, 10, :, :] = 1
    _, (d_col_ou, d_surf_ou) = torch.autograd.functional.jvp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    _, (d_col_in, d_surf_in) = torch.autograd.functional.vjp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    print('ADJ test :', torch.sum(d_col_ou * pert) -    torch.sum(d_col_in * pert), torch.sum(d_col_in * pert),  torch.sum(d_col_ou * pert))


# %% Test on cube deconstruction
import os

plt.imshow(pert[0,:,:,0,0].to('cpu').numpy()); plt.title('Perturbation')
plt.savefig(f'{graph_path}perturbation.jpg')
plt.show();plt.close('all')

###############################
plt.imshow(d_surf_ou[0,:,:,0].to('cpu').numpy());
plt.colorbar(); plt.title(f"Output_pert, {DL_train.surface_vars[0]}")
plt.savefig(f'{graph_path}tlm_col_output.jpg')
plt.show();plt.close('all')

###############################
plt.imshow(d_surf_in[0,:,:,0].to('cpu').numpy());
plt.colorbar(); plt.title(f"Input_pert, {DL_train.surface_vars[0]}")
plt.savefig(f'{graph_path}adj_col_output.jpg')
plt.show();plt.close('all')

plt.imshow(forced_t1[0,:,:,3].to('cpu').numpy());
plt.colorbar(); plt.title(f"Sunlight")
plt.savefig(f'{graph_path}solar.jpg')
plt.show();plt.close('all')

plt.imshow(surf_pred[0,:,:,-1].to('cpu').detach().numpy())
plt.colorbar(); plt.title(f"Input predicted {DL_train.surface_vars[0]}")
plt.savefig(f'{graph_path}surf_prediction.jpg')
plt.show();plt.close('all')

# Quick ADJ test:
with torch.no_grad():
    forecaster_for_grad = lambda x, y: forecaster(x, y, forced_t1[[0]])
    pert =  torch.randn_like(col_t1[[0]])
    _, (d_col_ou, d_surf_ou) = torch.autograd.functional.jvp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    _, (d_col_in, d_surf_in) = torch.autograd.functional.vjp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    print('ADJ test :', torch.sum(d_col_ou * pert) -    torch.sum(d_col_in * pert), torch.sum(d_col_in * pert),  torch.sum(d_col_ou * pert))

# %% End
print('Hello World')
