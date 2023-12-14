from src import DataLoader
from src.utils import Forcing_Generator
from src.forecaster import Forecaster, Physics, Normalizer
from src.dycore import Dycore

import yaml

with open('test_parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

data_path = parameters['data_path']
surface_vars = parameters['variables']['surface']
column_vars = parameters['variables']['column']
# Dycore needs: u, v, phis

DL = DataLoader(data_path, surface_vars, column_vars)
lats, lons = DL.get_lat_lon()

FG = Forcing_Generator(lats, lons, 1)
(surf, col), (surf_t2, col_t2), t  = DL[0]
forced = FG.generate(t)


input_normalizer = {'surface':Normalizer(0,1), 'column':Normalizer(0,1), 'forced':Normalizer(0,1)}
output_normalizer = {'surface':Normalizer(0,1), 'column':Normalizer(0,1)}

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
optimizer = torch.optim.SGD(forecaster.parameters(), lr=0.001, momentum=0.9)
L = torch.nn.MSELoss()

# DATA GEN:
(surf, col), (surf_t2, col_t2), t  = DL[0]
forced = FG.generate(t)
### Reset optim
optimizer.zero_grad()
### Forward
col2, surf2 = forecaster.forward(col, surf, forced)
l1 =  L(col_t2, col2)
l2 =  L(surf_t2, surf2)
loss = l1 + l2
loss.requires_grad=True
loss.backward()

print('Ok')

