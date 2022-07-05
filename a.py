import numpy as np
import cv2 as cv


img = cv.imread('C:/datasets/MPI-Sintel/MPI-Sintel-complete/training/final/temple_3\\frame_0049.png')

cv.imshow('img', img)
cv.waitKey(0)