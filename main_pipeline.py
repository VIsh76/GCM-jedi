from src import DataLoader, Forcing_Generator
from src.forecaster import Forecaster, Physics, Normalizer
from src.model import Dycore
import numpy as np
import matplotlib.pyplot as plt
import yaml
import tqdm
import warnings
import torch

# %% Initialisation
torch.manual_seed(0)
warnings.warn

with open('test_parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

batch_size = 2
data_path = parameters['data_path']
pred_surface_vars = parameters['variables']['pred']['surface']
pred_column_vars = parameters['variables']['pred']['column']

cst_surface_vars = parameters['variables']['forced']['cst']
forced_surface_vars = parameters['variables']['forced']['initialized']

#%% Initilized dataloaders:
DL0 = DataLoader(data_path, 1, [], [], cst_surface_vars)
(_, _, forced), _, _ = DL0[0]
lats, lons = DL0.get_lat_lon()
FG = Forcing_Generator(2, lats, lons, forced)
del(DL0)
DL = DataLoader(data_path, batch_size, pred_surface_vars, pred_column_vars, forced_surface_vars)
(col_t1, surf_t1, forced_t1), (col_t2, surf_t2, _), t  = DL[0]
forced = FG.generate(t, forced_t1)


# %% Architecture :
norm_path = 'data/norms'
input_normalizer = {'column':Normalizer(np.load('data/norms/input_means_col.npy'), np.load('data/norms/input_std_col.npy')), 
                    'surface' :Normalizer(np.load('data/norms/input_means_sur.npy'), np.load('data/norms/input_std_sur.npy'))}
                    #'forced' :Normalizer(0, 1)}
output_normalizer = {'column':Normalizer(0*np.load('data/norms/output_means_col.npy'), 1+0*np.load('data/norms/output_std_col.npy')), 
                    'surface'  :Normalizer(*np.load('data/norms/output_means_sur.npy'), 1+0*np.load('data/norms/output_std_sur.npy'))}

from src.model import fill_missing_values
parameters = fill_missing_values(parameters)
physic_nn = Physics(parameters=parameters)

dcore = Dycore()
forecaster = Forecaster(dcore, physic_nn, input_normalizer, output_normalizer, dt=1)
optimizer = torch.optim.SGD(forecaster.parameters(), lr=0.000001, momentum=0.9)
L_avg = torch.nn.MSELoss(reduction='mean')

from src.utils.nn import number_of_parameters
print("Number of parameters of the forecaster : \t", number_of_parameters(forecaster))
print("Number of parameters of the physic : \t", number_of_parameters(forecaster.physic))
print("Number of parameters of the encoder : \t", number_of_parameters(forecaster.physic.encoder))
print("Number of parameters of the decoder : \t", number_of_parameters(forecaster.physic.decoder))
print("Number of parameters of the processor : \t", number_of_parameters(forecaster.physic.processor))
print("Number of parameters of the embedding sur : \t", number_of_parameters(forecaster.physic.sur_embedding))
print("Number of parameters of the embedding col : \t", number_of_parameters(forecaster.physic.col_embedding))


# %% Training test
(col_t1, surf_t1, forced_t1), (col_t2, surf_t2, _), t  = DL[0]
forced = FG.generate(t, forced_t1)

avg_pred = []
losses_items_avg = []
for d in tqdm.tqdm(range(1)):
    ### Reset optim
    optimizer.zero_grad()
    ### Forward
    col_pred, surf_pred = forecaster.forward(col_t1, surf_t1, forced)
    l1 =  L_avg(col_pred , col_t2)
    l2 =  L_avg(surf_pred, surf_t2)
    loss = l1 + l2
    loss.backward()
    torch.nn.utils.clip_grad_norm_(forecaster.parameters(),1000)
    optimizer.step()

    losses_items_avg.append(loss.item())
    avg_pred.append(surf_pred.mean().item())

print('Loss avg:', losses_items_avg)
print(surf_t1.shape, surf_t1.mean(), surf_pred.mean())
print(torch.std(col_t1 - col_t2), torch.std(surf_t1 - surf_t2))
plt.plot(losses_items_avg)
#plt.show()


# %% Tests on gradients:
with torch.no_grad():
    forecaster_for_grad = lambda x, y: forecaster(x, y, forced[[0]])
    pert = col_t1[[0]] * 0
    pert[0, 100, :, 0] = 1
    _, (d_col_ou, d_surf_ou) = torch.autograd.functional.jvp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )
    _, (d_col_in, d_surf_in) = torch.autograd.functional.vjp(forecaster_for_grad, (col_t1[[0]], surf_t1[[0]]), (pert, surf_t1[[0]]*0) )


# %% Test on cube deconstruction

from src.analysis.rebuild import deconstruct_cube, flat_to_cube_sphere

Y = flat_to_cube_sphere(pert.numpy())
V = deconstruct_cube(Y[0,0,:,:,:,0])
plt.imshow(V)
plt.show()

Y = flat_to_cube_sphere(d_surf_ou.numpy())
V = deconstruct_cube(Y[0,:,:,:,0])
plt.imshow(V)
plt.show()

Y = flat_to_cube_sphere(d_surf_in.numpy())
V = deconstruct_cube(Y[0,:,:,:,0])
plt.imshow(V)
plt.show()

Y = flat_to_cube_sphere(forced.numpy())
V = deconstruct_cube(Y[0,:,:,:,-1])
plt.imshow(V)
plt.show()

Y = flat_to_cube_sphere(forced.numpy())
V = deconstruct_cube(Y[0,:,:,:,4])
plt.imshow(V)
plt.show()

# %% End
print('Hello World')
