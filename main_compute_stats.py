from src.load_data.renorm import compute_renorm_input
from src import DataLoader

import yaml
import numpy as np

with open('test_parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)

data_path = parameters['data_path']
surface_vars = parameters['variables']['pred']['surface']
column_vars = parameters['variables']['pred']['column']
# Dycore needs: u, v, phis

DL = DataLoader(data_path, 1, surface_vars, column_vars, [])
norm_dict = compute_renorm_input(DL, 10, {}, {})

for data_name in norm_dict:
    np.save(f'data/norms/{data_name}', norm_dict[data_name])
