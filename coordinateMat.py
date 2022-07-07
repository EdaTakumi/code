import numpy as np


def coordinateMat(size):
    coord_x = np.zeros(size, dtype=np.float32)
    coord_y = np.zeros(size, dtype=np.float32)
    for y in range(0, size[0], 1):
        for x in range(0, size[1], 1):
            coord_x[y][x] = x
            coord_y[y][x] = y
            
    return coord_x, coord_y


if __name__ == '__main__':
    coordinateMat()
