from torch import nn
import torch

class Normalizer(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalizer, self).__init__()
        self.mean = torch.from_numpy(mean).to(device=device) 
        self.std = torch.from_numpy(std).to(device=device)
        
    def norm(self, x):
        return (x - self.mean) / self.std 
    
    def unnorm(self, x):
        return (x * self.std + self.mean) 
