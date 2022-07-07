import numpy as np
import cv2 as cv

imgs = []

img0 = cv.imread('C:/datasets/MPI-Sintel/MPI-Sintel-complete/training/final/temple_3\\frame_0049.png')

imgs.append(img0)

img1 = cv.imread('C:/datasets/MPI-Sintel/MPI-Sintel-complete/training/final/temple_3\\frame_0048.png')

imgs.append(img1)

imgs = np.array(imgs)

cv.imshow('img', imgs[0])
cv.waitKey(0)