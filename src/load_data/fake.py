import numpy as np

lat = 10
lon = 10
lev = 10

def load_epoch(i, lat, lon, lev):
    return np.zeros((10_000, lat, lon, lev))
