import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime

id_var=5

def update_col(frame, ax, id_var, lev, DL, **kwarg):
    L, t_list = DL[frame]
    (col_t1, surf_t1, forced_t1) = L[0]
    t = datetime.datetime.strftime(t_list[0][0], DL.time_format)
    ax.set_title(t)
    ln = ax.imshow(col_t1.to('cpu').numpy()[0,:,:,lev, id_var])
    return ln

def update_delta_col(frame, ax, vmax, id_var, lev, DL, **kwarg):
    L, t_list = DL[frame, 1]
    (col_t1, surf_t1, forced_t1) = L[0]
    (col_t2, surf_t2, forced_t1) = L[1]
    t = datetime.datetime.strftime(t_list[0][0], DL.time_format)
    ax.set_title(t)
    ln = ax.imshow( (col_t1 - col_t2).to('cpu').numpy()[0,:,:,lev, id_var], vmax=vmax, vmin=-vmax, cmap='coolwarm')
    return ln

def update_force(frame, ax, id_var, vmax, DL, FG, **kwarg):
    L, t_list = DL[frame]
    (col_t1, surf_t1, forced_t1) = L[0]
    forced_t1 = FG.generate(t_list[0], forced_t1)
    t = datetime.datetime.strftime(t_list[0][0], DL.time_format)
    ax.set_title(t)
    ln = ax.imshow( (forced_t1).to('cpu').numpy()[0,:,:, id_var], cmap='coolwarm', vmax=vmax, vmin=-vmax)
    return ln

def generate_animation(update_fct, kwarg_fct, kwarg_ani, add_reset=True, lat=1, lon=1):
    fig, ax = plt.subplots()
    if add_reset:
        kwarg_ani['frames'][-1] = -1
    def lambda_fct(frame):
        if frame>=0:
            return update_fct(frame=frame, ax=ax, **kwarg_fct)
        else:
            ax.set_title('reset')
            return ax.imshow(np.zeros((lat, lon)))
    ani = FuncAnimation(fig, lambda_fct, **kwarg_ani)
    return ani
