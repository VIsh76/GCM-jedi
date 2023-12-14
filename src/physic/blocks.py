from torch import nn

class MLPBlock(nn.Module):    
    def __init__(self, in_channels, out_channels, activation:bool) -> None:
        super().__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        self.activation = activation
        self.act = nn.SELU()
 
    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            return self.act(x)
        else:
            return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation:bool) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.activation = activation
        self.act = nn.SELU()
 
    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            return self.act(x)
        else:
            return x
