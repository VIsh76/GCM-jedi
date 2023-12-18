import torch
from .load_file import DataLoader

def compute_renorm_input(dataloader:DataLoader,length, column_var:dict, surface_var:dict):
    # Inputs
    input_std_col   = torch.zeros( (1, 1, dataloader.n_levels, len(dataloader.column_vars)))
    input_mean_col = torch.zeros( (1, 1, dataloader.n_levels, len(dataloader.column_vars)))
    input_std_sur   = torch.zeros( (1, 1, len(dataloader.surface_vars)))
    input_mean_sur = torch.zeros( (1, 1, len(dataloader.surface_vars)))

    # Output:
    output_std_col   = torch.ones( (1, 1, dataloader.n_levels, len(dataloader.column_vars)))
    output_mean_col = torch.zeros( (1, 1, dataloader.n_levels, len(dataloader.column_vars)))
    output_std_sur   = torch.ones( (1, 1, len(dataloader.surface_vars)))
    output_mean_sur = torch.zeros( (1, 1, len(dataloader.surface_vars)))

    # Means
    print('Compute mean')
    for l in range(length):
        (surface_x, column_x),  (surface_y, column_y), t = dataloader[l]
        input_mean_col +=  torch.sum(column_x, axis=(0, 1, 2), keepdims=True)
        input_mean_sur +=  torch.sum(surface_x, axis=(0, 1), keepdims=True)
        output_mean_col += torch.sum(column_y - column_x, axis=(0, 1, 2), keepdims=True)
        output_mean_sur += torch.sum(surface_y - surface_x, axis=(0, 1), keepdims=True)
    
    n_terms = length * dataloader.batch_size * dataloader.n_lats * dataloader.n_lons
    input_mean_col  /= (n_terms * dataloader.n_levels) 
    input_mean_sur  /= n_terms
    output_mean_col /= (n_terms * dataloader.n_levels) 
    output_mean_sur /= n_terms

    # Stds
    print('Compute std')
    for l in range(length):
        (surface_x, column_x),  (surface_y, column_y), t = dataloader[l]
        input_std_col += torch.sum(   (input_mean_col - column_x)**2, axis=(0, 1, 2), keepdims=True)
        input_std_sur += torch.sum(   (input_mean_sur - surface_x)**2, axis=(0, 1), keepdims=True)
        output_mean_col += torch.sum( (output_mean_col - (column_y - column_x))**2, axis=(0, 1, 2), keepdims=True)
        output_mean_sur += torch.sum( (output_mean_sur - (surface_y - surface_x))**2, axis=(0, 1), keepdims=True)
    
    input_std_col  /= (n_terms * dataloader.n_levels) 
    input_std_sur  /= n_terms
    output_std_col /= (n_terms * dataloader.n_levels) 
    output_std_sur /= n_terms

    return {'input_std_col':input_std_col, 
            'input_means_col':input_mean_col, 
            'input_std_sur':input_std_sur, 
            'input_means_sur':input_mean_sur, 
            'output_std_col':output_std_col, 
            'output_means_col':output_mean_col, 
            'output_std_sur':output_std_sur, 
            'output_means_sur':output_mean_sur}

# 1/N 
# SQRT SUM( (MEAN - VAL)**2 )
