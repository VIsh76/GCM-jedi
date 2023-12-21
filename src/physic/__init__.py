from .coders import Encoder, Decoder
from .process import Process
from .embedding import NoEmbedding, Column_Embedding, Surface_Embedding
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
        embd_surface_dim = 8
        embd_column_dim = 32
        self.sur_embedding = Surface_Embedding(3, surface_vars_input, 16, embd_surface_dim)
        self.col_embedding = Column_Embedding(3, column_vars_input, 64, embd_column_dim, kernel_size=5)
        self.encoder   = Encoder(surface_vars=embd_surface_dim,
                                 n_levels=n_levels,
                                 column_vars=embd_column_dim,
                                 features=hidden_dims)
        self.processor = Process(n_layer, hidden_dims, hidden_dims, hidden_dims)
        self.decoder   = Decoder(surface_vars=surface_vars_output,
                                 n_levels=n_levels,
                                 column_vars=column_vars_output,
                                 features=hidden_dims)

    def forward(self, surface, col_var, forced):
        all_surface = torch.concat([surface, forced], dim=-1)
        sur_embd = self.sur_embedding.forward(all_surface)
        col_embd = self.col_embedding.forward(col_var)

        x = self.encoder.forward(sur_embd, col_embd)
        x = self.processor.forward(x)
        surface, col = self.decoder.forward(x)
        return surface, col
    