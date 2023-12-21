from src import DataLoader, Forcing_Generator
from src.forecaster import Forecaster, Physics, Normalizer
from src.dycore import Dycore
import matplotlib.pyplot as plt
import yaml
import tqdm
import warnings

warnings.warn

with open('test_parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

data_path = parameters['data_path']
surface_vars = parameters['variables']['surface']
column_vars = parameters['variables']['column']

# Dycore needs: u, v, phis
DL = DataLoader(data_path, surface_vars, column_vars)
lats, lons = DL.get_lat_lon()

FG = Forcing_Generator(lats, lons, 2)
(surf, col), (surf_t2, col_t2), t  = DL[0]
forced = FG.generate(t)

import numpy as np

norm_path = 'data/norms'

input_normalizer = {'column':Normalizer(np.load('data/norms/input_means_col.npy'), 1+0*np.load('data/norms/input_std_col.npy')), 
                    'surface' :Normalizer(np.load('data/norms/input_means_sur.npy'),1+0* np.load('data/norms/input_std_sur.npy'))}
                    #'forced' :Normalizer(0, 1)}
output_normalizer = {'column':Normalizer(np.load('data/norms/output_means_col.npy'), 1+0*np.load('data/norms/output_std_col.npy')), 
                    'surface'  :Normalizer(np.load('data/norms/output_means_sur.npy'), 1+0*np.load('data/norms/output_std_sur.npy'))}

physic_nn = Physics(surface_vars_input  = len(surface_vars) + len(FG),
                    column_vars_input   = len(column_vars),
                    surface_vars_output = len(surface_vars),
                    column_vars_output  = len(column_vars),
                    n_layer=3, 
                    n_levels=DL.n_levels,
                    hidden_dims=10
                    )
import torch

dcore = Dycore()
forecaster = Forecaster(dcore, physic_nn, input_normalizer, output_normalizer, dt=15)
optimizer = torch.optim.SGD(forecaster.parameters(), lr=0.000001, momentum=0.9)
L = torch.nn.MSELoss(reduce='mean')

# DATA GEN:
(surf, col), (surf_t2, col_t2), t  = DL[0]
forced = FG.generate(t)
losses_items = []
for d in tqdm.tqdm(range(10)):
    ### Reset optim
    optimizer.zero_grad()
    ### Forward
    col2, surf2 = forecaster.forward(col, surf, forced)
    l1 =  L(col_t2, col2)
    l2 =  L(surf_t2, surf2)
    loss = l1 + l2
    loss.backward()
    torch.nn.utils.clip_grad_norm_(forecaster.parameters(),1000)
    optimizer.step()
    losses_items.append(loss.item())

print('Loss :', losses_items)
#plt.plot(losses_items)
print('Ok')
