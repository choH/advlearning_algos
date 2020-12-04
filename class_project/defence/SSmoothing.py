import numpy as np
import math

def kernel_array(data, xi, yj, nelx, nely, r, N):
    arr = np.zeros((N, N))
    startx = xi - r
    starty = yj - r
    for i in range(N):
        for j in range(N):
            x_i = startx + i
            y_i = starty + j
            if x_i < 0 or y_i < 0 or x_i >= nelx or y_i >= nely:
                continue
            arr[i][j] = data[x_i][y_i]
    return arr

def create_kernel(rmin):
    N = 2*math.floor(rmin) + 1
    H = np.zeros((N, N))
    center_x = N/2
    center_y = N/2
    # print(center_x, center_y)
    for i in range(N):
        for j in range(N):
            H[i][j] = max(0, math.sqrt((center_x - i)*(center_x - i)+(center_y - j)*(center_y - j)))
    H= H/np.sum(H)
    return H

def SpatialS(x_art, window_size, way):
    kernel = np.ones((window_size, window_size))
    kernel = kernel/float(window_size*window_size)
    rmin = int(window_size/2)
    if way=="rmin":
        # kernel = create_kernel(window_size/2)
        kernel2 = np.zeros((window_size, window_size))
        kernel2[rmin][rmin] = 1.0 # sharpen
        kernel = 2 * kernel2 - kernel
    channal, NX, NY = x_art.shape
    x_art_copy = x_art.copy()
    for k in range(channal):
        for i in range(NX):
            for j in range(NY):
                val_ = kernel_array(x_art[k], i, j, NX, NY, rmin, window_size)
                if way=="0.5mean":
                    x_art_copy[k][i][j] = np.sum((val_**0.5*kernel))**2
                else:
                    x_art_copy[k][i][j] = np.sum(val_*kernel)
    return x_art_copy


















