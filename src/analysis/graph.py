import matplotlib.pyplot as plt
import numpy as np


def plot_var_profiles(t1, t2, pred):
    """
    t1, t2, pred size [x,y]
    """
    f = plt.figure(figsize=(40, 10))
    titles = ['t1', 't2', 'pred']
    for i, data_0 in enumerate([t1,t2,pred]):
        ax = f.add_subplot(1, 5, i+1)
        ax.set_title(titles[i])
        ax.imshow(data_0)
    
    data1 = t1 - t2
    data2 = pred - t2
    vmax = max(data1, data2)
    ax = f.add_subplot(5, 1, 4)
    ax.imshow(data1, cmap='coolwarm', vmax=vmax, vmin=-vmax);
    ax.set_title
    
    ax = f.add_subplot(5, 1, 5)
    ax.imshow(data2, cmap='coolwarm', vmax=vmax, vmin=-vmax);
    plt.colorbar()
    return f


