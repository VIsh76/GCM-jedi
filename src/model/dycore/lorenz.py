from torch import nn
import torch

F = 8
dt = 0.1

class Lorenz(nn.Module):
    # Input: [lat, lon, lev, v]
    # -> Apply FV3 on longitude
    def __init__(self, *args, **kwargs) -> None:
        super(Lorenz, self).__init__(*args, **kwargs)
    
    @staticmethod
    def forward(input:torch.Tensor, dt=dt):
        return input + dt * Lorenz.derivative(input)
    
    @staticmethod
    def derivative(input : torch.Tensor):
        input1 = torch.roll(input, shifts=1, dims=1)
        input_1 = torch.roll(input, shifts=-1, dims=1)
        input_2 = torch.roll(input, shifts=-2, dims=1)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        forced = torch.ones(input.size())
        forced[:, 1] += F
        # d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return (input1 - input_2) * input_1 - input + forced

    @staticmethod
    def grad(input:torch.Tensor):
        input1 = torch.roll(input, shifts=1, dims=1)
        input_1 = torch.roll(input, shifts=-1, dims=1)
        input_2 = torch.roll(input, shifts=-2, dims=1)
        dxi1    = input_1
        dxi     = - torch.ones_like(input1)
        dxi_1   = input1 - input_2
        dxi_2   = -input_1
        return dxi1, dxi, dxi_1, dxi_2
 
    @staticmethod
    def tlm(input, delta):
        dxi1, dxi, dxi_1, dxi_2 = Lorenz.grad(input)
        output = torch.zeros_like(input)
        output += dxi1 * torch.roll(delta, 1, dims=1)
        output +=  dxi * delta
        output += dxi_1 * torch.roll(delta, 1, dims=-1)
        output += dxi_2 * torch.roll(delta, 1, dims=-2)
        return output
