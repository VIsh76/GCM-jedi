from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, n_blocks, surface_vars, n_levels, column_vars, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.n_levels = n_levels 
        self.columns_vars = column_vars 
        self.surface_vars = surface_vars         
        self.input_size = n_levels * column_vars + surface_vars
        self.layer = nn.Linear(self.input_size, out_features=output_size)
        
    def forward(self, surface, col):
        x = torch.concatenate( [surface, torch.flatten(col, start_dim=-2, end_dim=-1) ], axis=-1)
        x = self.layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self,  n_blocks, surface_vars, n_levels, column_vars, hidden_size, input_size):
        super(Decoder, self).__init__()
        self.n_levels = n_levels 
        self.columns_vars = column_vars 
        self.surface_vars = surface_vars 
        self.input_size = input_size
        
        self.output_size = n_levels * column_vars + surface_vars
        self.layer = nn.Linear(self.input_size, out_features=self.output_size)
      
    def forward(self, x):
        x = self.layer(x)
        splits = torch.split(x, split_size_or_sections=self.surface_vars, dim=-1)
        surf = splits[0]
        col = torch.concatenate( splits[1:], dim=-1)
        col = torch.reshape(col, [*col.size()[:-1],  self.n_levels, self.columns_vars])
        return surf, col

    