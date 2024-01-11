import numpy as np

def flat_to_cube_sphere(X, nf=6):
    """
    Given an array of size (bs, lat*lon*nf, nb_var)
    Returns an array of size (bs, nf, lat, lon, nb_var)
    Given an array of size (bs, lat*lon*nf, lev, nb_var)
    Returns an array of size (bs, lev, nf, lat, lon, nb_var)
    
    """
    bs, h, nb_var = X.shape
    lat_square = h / nf
    lat = int( np.sqrt(lat_square))
    lon = lat
    Y = np.moveaxis(X, 1, -1)
    Y = np.reshape(Y, (bs, nb_var, nf, lat, lon))
    Y = np.moveaxis(Y, 1, -1)
    return Y

def latlon_to_cube_sphere(X, nf=6):
    """
    Given an array of size (bs, lat, lon*nf, nb_var)
    Returns an array of size (bs, nf, lat, lon, nb_var)
    Given an array of size (bs, lat, lon*nf, lev, nb_var)
    Returns an array of size (bs, lev, nf, lat, lon, nb_var)
    
    """
    bs, lat, lon_nf, nb_var = X.shape
    nf = lon_nf // lat
    lon = lat
    Y = np.moveaxis(X, -1, 1)
    Y = np.reshape(Y, (bs, nb_var, lat, nf, lon))
    Y = np.moveaxis(Y, -2, 1) # nf at the start
    Y = np.moveaxis(Y, 2, -1) # var at the end
    return Y


def reshaper(data, prec=48):
    return np.reshape(data, (prec, prec, 6)).T

def set_ij(X, x,i,j, dim):
    X[i*dim:(i+1)*dim,  j*dim:(j+1)*dim]=x

def deconstruct_cube(data):
    """Given a data file, reconstruct a bigger matrix that emcopass the entire earth flat

    Args:
        data (np.array): size (6, prec, prec)

    Returns:
        np.array: matrix of size (prec*3, prec*4) showing all faces of the cube
    """
    x = data.shape[-1]
    X_img = np.nan + np.zeros((x*3, x*4))
    X_img[:x, :x] = data[0].T
    set_ij(X_img, data[0].T, 0, 0, dim=x) # South Atl
    set_ij(X_img, data[1].T, 1, 0, dim=x) # Asia
    set_ij(X_img, data[2].T, 1, 1, dim=x) # NPole
    set_ij(X_img, np.flip(data[4], axis=0), 1, 2, dim=x) # America
    set_ij(X_img, np.flip(data[5], axis=0), 1, 3, dim=x) # South pole 
    set_ij(X_img, np.flip(data[3], axis=1), 2, 0, dim=x) # Australia
    return X_img
