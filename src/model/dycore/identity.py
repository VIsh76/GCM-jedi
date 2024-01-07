from torch import nn

class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

    def __call__(self, x):
        return x
