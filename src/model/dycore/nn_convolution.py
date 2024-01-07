import torch
from torch import nn
from torch.nn import Conv3d


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation:bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size, 1), padding='same')
        self.activation = activation
        self.act = nn.SELU()
 
    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            return self.act(x)
        else:
            return x
 

class NN_Dycore(nn.Modules):
    def __init__(self, n_blocks, input_var, hidden_var, kernel_size) -> None:
        self.convolution = Conv3d(in_channels=input_var, out_channels=input_var, kernel_size=(1,2,2), padding='same')

    def forward(x):
        return x


class Column_Embedding(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, output_size, kernel_size):
        super(Column_Embedding, self).__init__()
        self.blocks = []
        self.blocks.append( Conv2DBlock(input_size, hidden_size, kernel_size, activation=True) )
        for _ in range(n_blocks-2):
            self.blocks.append(  Conv2DBlock(hidden_size, hidden_size, kernel_size, activation=True) )
        self.blocks.append(  Conv2DBlock(hidden_size, output_size, kernel_size, activation=False) )
        for i in range(len(self.blocks)):
            self.add_module(f"col_embd_block_{i}", self.blocks[i])


    def forward(self, x):
        # Input size is (bs, horizontal, lev, nb_vars) -> Reshape
        # Pytorch uses convolution differently where the variable dimension is not at the end but at the start!
        x = torch.swapaxes(x, -1, 1) # Switch (bs, profile_points, lev, vars) to  (bs, vars, lev, profile_points)
        # Perform the kernels
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        # Put the dimension at the end
        x = torch.swapaxes(x, -1, 1) # Switch(bs, vars, lev, profile_points) to (bs, profile_points, lev, vars) 
        return x
