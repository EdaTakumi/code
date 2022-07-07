import cv2 as cv
import numpy as np


def coordinateMat(size):
    coord_x = np.zeros(size, dtype = np.float32)
    coord_y = np.zeros(size, dtype = np.float32)
    for y in range(0, size[0], 1):
        for x in range(0, size[1], 1):
            coord_x[y][x] = x
            coord_y[y][x] = y
    return coord_x, coord_y


img1 = cv.imread("C:/datasets/MPI-Sintel/MPI-Sintel-complete/training/final/alley_1/frame_0001.png")
img2 = cv.imread("C:/datasets/MPI-Sintel/MPI-Sintel-complete/training/final/alley_1/frame_0002.png")

flow = cv.readOpticalFlow("C:/datasets/MPI-Sintel/MPI-Sintel-complete/training/flow/alley_1/frame_0001.flo")

flow_x, flow_y = cv.split(flow)

coord_x, coord_y = coordinateMat(img1.shape[:2])

map_x = coord_x + flow_x
map_y = coord_y + flow_y

dst = cv.remap(img2, map_x, map_y, cv.INTER_LINEAR)
cv.imshow('dst', dst)

psnr = cv.PSNR(dst, img1)

gray_dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

img_diff = cv.absdiff(gray_dst, gray_img1)
cv.imshow('img_diff', img_diff)
cv.waitKey(0)
