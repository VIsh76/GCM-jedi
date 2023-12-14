import torch

def Loss(x, y):
    return torch.nn.MSELoss(x, y)

