from torch import nn
from.blocks import ConvBlock, MLPBlock

class NoEmbedding(nn.Module):
    def __init__(self):
        super(NoEmbedding, self).__init__()
    
    def forward(self,x):
        return x

class Surface_Embedding(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, output_size):
        super(Surface_Embedding, self).__init__()
        self.blocks = []
        self.blocks.append( MLPBlock(input_size, hidden_size, activation=True) )
        for _ in range(n_blocks-2):
            self.blocks.append(  MLPBlock(hidden_size, hidden_size, activation=True) )
        self.blocks.append(  MLPBlock(hidden_size, output_size, activation=False) )
        

class Vertical_Embedding(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, output_size, kernel_size):
        super(Vertical_Embedding, self).__init__()
        self.blocks = []
        self.blocks.append( ConvBlock(input_size, hidden_size, kernel_size, activation=True) )
        for _ in range(n_blocks-2):
            self.blocks.append(  ConvBlock(hidden_size, hidden_size, kernel_size, activation=True) )
        self.blocks.append(  ConvBlock(hidden_size, output_size, kernel_size, activation=False) )

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x
    
