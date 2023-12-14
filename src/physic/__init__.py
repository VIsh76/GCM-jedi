from .coders import Encoder, Decoder
from .process import Process
from .embedding import NoEmbedding
from .normalizer import Normalizer
from torch import nn
import torch


class Physics(nn.Module):
    def __init__(self, surface_vars_input, 
                       column_vars_input, 
                       n_levels, 
                       surface_vars_output,
                       column_vars_output,
                       n_layer,
                       hidden_dims):
        super(Physics, self).__init__()
        self.encoder   = Encoder(surface_vars=surface_vars_input,
                                 n_levels=n_levels,
                                 column_vars=column_vars_input,
                                 features=hidden_dims)
        self.processor = Process(n_layer, hidden_dims, hidden_dims, hidden_dims)
        self.decoder   = Decoder(surface_vars=surface_vars_output,
                                 n_levels=n_levels,
                                 column_vars=column_vars_output,
                                 features=hidden_dims)
        self.sur_embedding = NoEmbedding()
        self.col_embedding = NoEmbedding()

    def forward(self, surface, col_var, forced):
        all_surface = torch.concat([surface, forced], dim=-1)
        sur_embd = self.sur_embedding(all_surface)
        col_embd = self.col_embedding(col_var)

        x = self.encoder.forward(sur_embd, col_embd)
        x = self.processor.forward(x)
        surface, col = self.decoder.forward(x)
        return surface, col
    