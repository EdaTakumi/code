import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def drawVector(flow, figsize, per_pixel=20):
    height, width = flow.shape[0], flow.shape[1]
    plt.figure(figsize = figsize)

    for y in range(0, height, per_pixel):
        for x in range(0, width, per_pixel):
            u, v = flow[y][x]
            plt.quiver(x, height-1-y, 5*u, 5*v, width=0.003, headwidth=3, headlength=5, color='purple', angles='xy', scale_units='xy', scale=1.0)
    
    plt.xlim([0, width])
    plt.ylim([0, height])
    plt.show()


if __name__ == '__main__':
    drawVector()
    