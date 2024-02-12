import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime

def update_col(frame, fig, ax, id_var, lev, DL, add_colobar=False, cax=None, vmin=None, vmax=None, **kwarg):
    (col_t1, _, _), _, t_list = DL[frame]
    t = datetime.datetime.strftime(t_list[0], DL.time_format)
    data = col_t1.to('cpu').numpy()[0,:,:,lev, id_var]
    ax.set_title(t)
    if vmin is None and vmax is None:    
        im = ax.imshow(data, cmap='viridis')
    else:
        im = ax.imshow(data, vmax=vmax, vmin=vmin, cmap='viridis')        
    if add_colobar:
        fig.colorbar(im, cax)
    return im

def update_delta_col(frame, fig, ax, vmax, id_var, lev, DL,  add_colobar=False, cax=None, **kwarg):
    (col_t1, surf_t1, forced_t1), (col_t2, surf_t2, forced_t2), t_list = DL[frame]
    t = datetime.datetime.strftime(t_list[0], DL.time_format)
    ax.set_title(t)
    im = ax.imshow( (col_t2 - col_t1).to('cpu').numpy()[0,:,:,lev, id_var], vmax=vmax, vmin=-vmax, cmap='coolwarm')
    if add_colobar:
        fig.colorbar(im, cax)
    return im

def update_force(frame, fig, ax, id_var, vmax, DL, FG, add_colobar=False, cax=None, **kwarg):
    (_, _, forced_t1), _, t_list = DL[frame]
    t = datetime.datetime.strftime(t_list[0], DL.time_format)
    forced_t1 = FG.generate(t, forced_t1)
    ax.set_title(t)
    im = ax.imshow( (forced_t1).to('cpu').numpy()[0,:,:, id_var], cmap='coolwarm', vmax=vmax, vmin=-vmax)
    if add_colobar:
        fig.colorbar(im, cax)
    return im

def generate_animation(update_fct, kwarg_fct, kwarg_ani, add_reset=True,  add_colobar=False, lat=1, lon=1):
    fig, ax = plt.subplots()
    if add_reset:
        kwarg_ani['frames'][-1] = -1
    if add_colobar:
        im = ax.imshow(np.zeros((lat, lon)))
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
    else:
        cax = None
    def lambda_fct(frame):
        if frame>=0:
            return update_fct(frame=frame, fig=fig, ax=ax, add_colobar=add_colobar, cax=cax, **kwarg_fct)
        else:
            ax.set_title('reset')
            return ax.imshow(np.zeros((lat, lon)))
    ani = FuncAnimation(fig, lambda_fct, **kwarg_ani)
    return ani