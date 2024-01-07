from .coders import Encoder, Decoder
from .process import Process
from .embedding import NoEmbedding, Column_Embedding, Surface_Embedding
from .normalizer import Normalizer
from torch import nn
import torch

# embd_surface_dim = 8
# embd_column_dim = 32

class Physics(nn.Module):
    def __init__(self, 
                 sur_embedding:Surface_Embedding=None,
                 col_embedding:Column_Embedding=None,
                 encoder:Encoder=None,
                 decoder:Decoder=None,
                 processor:Process=None,
                 parameters={},
                 ) -> None:
        super().__init__()
        if len(parameters) > 0:
                self.sur_embedding = Surface_Embedding(**parameters['architecture']['surface_embedding'])
                self.col_embedding = Column_Embedding(**parameters['architecture']['column_embedding'])
                self.encoder = Encoder(**parameters['architecture']['encoder'])
                self.processor = Process(**parameters['architecture']['process'])
                self.decoder = Decoder(**parameters['architecture']['decoder'])
        else:
            check = sur_embedding is None or col_embedding is None or encoder is None or decoder is None or processor is None
            assert(not check)
            self.sur_embedding = sur_embedding
            self.col_embedding = col_embedding
            self.processor = processor
            self.encoder = encoder
            self.decoder = decoder


    def forward(self, surface, col_var, forced):
        all_surface = torch.concat([surface, forced], dim=-1)
        sur_embd = self.sur_embedding.forward(all_surface)
        col_embd = self.col_embedding.forward(col_var)

        x = self.encoder.forward(sur_embd, col_embd)
        x = self.processor.forward(x)
        surface, col = self.decoder.forward(x)
        return surface, col

