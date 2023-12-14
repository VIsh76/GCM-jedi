import torch
from torch import nn
from .dycore import Dycore
from .physic import Physics, Normalizer
from .utils import Forcing_Generator

class Forecaster(nn.Module):
    def __init__(self, dycore:Dycore, 
                       nn_physic:Physics, 
                       input_norm,
                       output_norm,
                       dt:float 
                       ):
        super(Forecaster, self).__init__()
        # Dynamical part:
        self.physic =nn_physic
        # Physical Part:
        self.dycore = dycore
        self.input_norm  = input_norm
        self.output_norm = output_norm
        self.dt = dt
        # Dimensions

    def ode(self, dx_phys, x_dyn):
        return x_dyn + dx_phys * self.dt

    def forward(self, var_column, var_surface, var_forced):
        d_phy_sur, d_phy_col = self.propagate_phy(var_column, var_surface, var_forced)
        d_dyn = self.propagate_dyn(var_column)
        o = self.ode(d_phy_col, d_dyn)
        return o, d_phy_sur

    def propagate_dyn(self, var_column):
        return self.dycore(var_column)
    
    def propagate_phy(self, var_column, var_surface, forced):
        # Norm:
        var_column = self.input_norm['column'].norm(var_column)
        var_surface = self.input_norm['surface'].norm(var_surface)
        forced = self.input_norm['forced'].norm(forced)
        # Process:
        surface, column = self.physic(var_surface, var_column, forced)
        # UnNorm:
        column = self.input_norm['column'].unnorm(var_column)
        surface = self.input_norm['surface'].unnorm(var_surface)
        return  surface, column
