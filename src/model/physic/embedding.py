from torch import nn
from.blocks import ConvBlock, MLPBlock
import torch

class NoEmbedding(nn.Module):
    def __init__(self):
        super(NoEmbedding, self).__init__()
    
    def forward(self,x):
        return x

class Surface_Embedding(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, output_size, **kwarg):
        super(Surface_Embedding, self).__init__()
        self.blocks = []
        self.blocks.append( MLPBlock(input_size, hidden_size, activation=True) )
        for _ in range(n_blocks-2):
            self.blocks.append(  MLPBlock(hidden_size, hidden_size, activation=True) )
        self.blocks.append(  MLPBlock(hidden_size, output_size, activation=False) )
        for i in range(len(self.blocks)):
            self.add_module(f"sur_embd_block_{i}", self.blocks[i])

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x
     

class Column_Embedding(nn.Module): # PADDING has to be circular
    def __init__(self, n_blocks, input_size, hidden_size, output_size, kernel_size, **kwarg):
        super(Column_Embedding, self).__init__()
        # For easier padding kernels should be odd numbers:
        #assert(kernel_size[0]%2==1 and kernel_size[1]%2==1 and kernel_size[2]%2==1)
        self.kernel_size = kernel_size
        self.paddings = 'same' #( (kernel_size[0]-1)//2, 0, (kernel_size[2]-1)//2) # lat, lon, lev; lon is circular
        self.blocks = []
        self.blocks.append( ConvBlock(input_size, hidden_size, kernel_size, activation=True, padding=self.paddings) )
        for _ in range(n_blocks-2):
            self.blocks.append(  ConvBlock(hidden_size, hidden_size, kernel_size, activation=True, padding=self.paddings) )
        self.blocks.append(  ConvBlock(hidden_size, output_size, kernel_size, activation=False, padding=self.paddings) )
        for i in range(len(self.blocks)):
            self.add_module(f"col_embd_block_{i}", self.blocks[i])
            
    def forward(self, x):
        # Input size is (bs, horizontal, lev, nb_vars) -> Reshape
        # Pytorch uses convolution differently where the variable dimension is not at the end but at the start!
        # Switch (bs, *profile_points, lev, vars) to  (bs, vars, *profile_points, lev)
        x = torch.moveaxis(x, -1, 1)
        k_lon_s = self.kernel_size[1] - 1
        # Perform the kernels
        for i in range(len(self.blocks)):
            # Upsampling blocks: The earth is not a square
            #lon_west = x[:, :, :, :k_lon_s//2] 
            #lon_east = x[:, :, :, -k_lon_s//2:]
            # Args:
            #x = torch.concatenate([lon_east, x, lon_west], axis=3)
            # Apply bloc (dimension [2, 3, 4] of input and output will be the same)
            x = self.blocks[i](x)
        # Put the variable dimension at the end
        x = torch.moveaxis(x, 1, -1) # Switch(bs, vars,  *profile_points, lev) to (bs, *profile_points, lev, vars) 
        return x