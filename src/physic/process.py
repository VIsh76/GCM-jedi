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

        for i in range(len(self.layers)):
            self.add_module(f"process_lin_{i}", self.layers[i])
        for j in range(len(self.act)):
            self.add_module(f"process_act_{j}", self.act[j])

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.act):
                x = self.act[i](x)
        return x
        