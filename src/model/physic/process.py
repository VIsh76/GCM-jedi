from torch import nn

class Process(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, output_size):
        super(Process, self).__init__()
        self.layers = []
        self.act = []
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.act.append(nn.SiLU())
        for _ in range(n_blocks-2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.act.append(nn.SiLU())
        self.layers.append(nn.Linear(hidden_size, output_size))   

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
        