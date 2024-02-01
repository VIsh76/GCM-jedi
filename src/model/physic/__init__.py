from .coders import Encoder, Decoder
from .process import Process
from .embedding import NoEmbedding, Column_Embedding, Surface_Embedding
from .normalizer import Normalizer
from torch import nn
import torch


class Physics(nn.Module):
    def __init__(self, 
                 col_embedding:Column_Embedding=None,
                 sur_embedding:Surface_Embedding=None,
                 encoder:Encoder=None,
                 decoder:Decoder=None,
                 processor:Process=None,
                 parameters={},
                 ) -> None:
        super().__init__()
        if len(parameters) > 0:
                self.col_embedding = Column_Embedding(**parameters['architecture']['column_embedding'])
                self.sur_embedding = Surface_Embedding(**parameters['architecture']['surface_embedding'])
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

    def forward(self, col_var, surface, forced):
        all_surface = torch.concat([surface, forced], dim=-1)
        sur_embd = self.sur_embedding.forward(all_surface)
        col_embd = self.col_embedding.forward(col_var)

        x = self.encoder.forward(col_embd, sur_embd)
        x = self.processor.forward(x)
        col, surface = self.decoder.forward(x)
        return col, surface


class Encode_Decode(nn.Module):
    def __init__(self, 
                 parameters,
                 *args, **kwargs,
                 ) -> None:
        super().__init__()
        self.encoder = Encoder(**parameters['architecture']['encoder'])
        #self.processor = Process(**parameters['architecture']['process'])
        self.decoder = Decoder(**parameters['architecture']['decoder'])


    def forward(self, col_var, surface, forced):
        all_surface = torch.concat([surface, forced], dim=-1)
        x = self.encoder.forward(col_var, all_surface)
        col, surface = self.decoder.forward(x)
        return col, surface

class LIN(nn.Module):
    def __init__(self, 
                 parameters,
                 *args, **kwargs,
                 ) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features=7, out_features=7)

    def forward(self, col, surface, forced):
        col = self.layer(col)
        return col, surface*0


