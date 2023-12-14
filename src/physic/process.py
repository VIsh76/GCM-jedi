from torch import nn

class Process(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dims, output_dims):
        super(Process, self).__init__()
        self.layers = []
        self.act = []
        self.layers.append(nn.Linear(input_dim, hidden_dims))
        self.act.append(nn.SiLU())
        for _ in range(n_layers-2):
            self.layers.append(nn.Linear(hidden_dims, hidden_dims))
            self.act.append(nn.SiLU())
        self.layers.append(nn.Linear(hidden_dims, output_dims))   
        # Last layer linear

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.act):
                x = self.act[i](x)
        return x
        