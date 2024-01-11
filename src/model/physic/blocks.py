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
    """
    Block to perform 1D or 3D convolution on the column:
    For 1D:
    To handle Pytorch gestion of convolution, since the horizontal variable is used
    The expected input size is (bs, var, horizontal, lev), so we apply a 2D convolution with a
    kernel size of 1 for the horizontal
    -------------
    For 3D:
    The expected input size is (bs, var, lat, lon, lev), so we apply a 3D convolution with 
    the chosen kernel sizes. The expected kernel size is for lev, lat, lon respectively
    """
    def __init__(self, in_channels, out_channels, kernel_size:int, activation:bool, one_d:bool) -> None:
        super().__init__()
        if one_d==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding='same')
        elif one_d==3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        else:
            assert(False)
        self.activation = activation
        self.act = nn.SELU()
 
    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            return self.act(x)
        else:
            return x


class AttentionHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)