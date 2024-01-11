import torch
from torch import nn
from .model import Dycore
from .model import Physics, Normalizer

class Forecaster(nn.Module):
    def __init__(self, dycore:Dycore, 
                       nn_physic:Physics, 
                       input_norm,
                       output_norm,
                       dt:float,
                       n_step_phys=1,
                       n_steps_dyn=1,
                       ):
        super(Forecaster, self).__init__()
        # Dynamical part:
        self.physic = nn_physic
        # Physical Part:
        self.dycore = dycore
        self.input_norm  = input_norm
        self.output_norm = output_norm
        self.n_step_phys = n_step_phys
        self.n_steps_dyn = n_steps_dyn
        assert(n_step_phys * n_steps_dyn > 0)
        self.dt = dt
        # Dimensions

    def ode(self, dx_phys, x_dyn):
        return x_dyn + dx_phys * self.dt

    def forward(self, var_column, var_surface, var_forced):
        d_dyn = self.propagate_dyn(var_column)
        d_phy_col, d_phy_sur = self.propagate_phy(var_column, var_surface, var_forced)
        col_ode = self.ode(d_phy_col, d_dyn)
        sur_ode = self.ode(d_phy_sur, var_surface)
        return col_ode, sur_ode

    def propagate_dyn(self, var_column):
        return self.dycore(var_column)
    
    def propagate_phy(self, var_column, var_surface, forced):
        # Norm:
        var_column = self.input_norm['column'].norm(var_column)
        var_surface = self.input_norm['surface'].norm(var_surface)
        #forced = self.input_norm['forced'].norm(forced)
        # Process:
        surface, column = self.physic(var_surface, var_column, forced)
        # UnNorm:
        column = self.output_norm['column'].unnorm(column)
        surface = self.output_norm['surface'].unnorm(surface)
        return  column, surface
