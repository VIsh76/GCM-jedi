import numpy as np

def lbd_scheduler(initial_lr:float, max_lr:float, epoch_init:int, epoch_decay:int):
    """ 
    Copies Graphcast scheduling lr
    Initialisation with increasing lr then half cos decay    
    """
    def f(epoch):
        if epoch < epoch_init:
            return initial_lr + epoch * (max_lr - initial_lr) / epoch_init 
        else:
            return np.cos(np.pi * (epoch - epoch_init) / 2 / epoch_decay) * max_lr
    return f
