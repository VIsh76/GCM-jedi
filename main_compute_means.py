from src.load_data.renorm import compute_renorm_input
from src import DataLoader
from src.utils import Forcing_Generator
from src.forecaster import Forecaster, Physics, Normalizer
from src.dycore import Dycore

import yaml
import numpy as np

with open('test_parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

data_path = parameters['data_path']
surface_vars = parameters['variables']['surface']
column_vars = parameters['variables']['column']
# Dycore needs: u, v, phis

DL = DataLoader(data_path, surface_vars, column_vars)
norm_dict = compute_renorm_input(DL, 10, {}, {})

for data_name in norm_dict:
    np.save(f'data/norms/{data_name}', norm_dict[data_name])
