import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

import mpi_sintel_datasets as msd


train_first_imgs, train_second_imgs, test_first_imgs, test_second_imgs, train_flows = msd.load_data()

print(len(train_first_imgs))
print(len(train_second_imgs))
print(len(train_flows))

cv.imshow('img', train_first_imgs[0])
cv.waitKey(0)

# x_train_first_img = train_first_imgs.reshape(len(train_first_imgs), train_first_imgs[0].shape[0], train_first_imgs[0].shape[1], 1) / 255
# x_train_second_img = train_second_imgs.reshape(len(train_second_imgs), train_second_imgs[0].shape[0], train_second_imgs[0].shape[1], 1) / 255

# x_test_first_img = test_first_imgs.reshape(len(test_first_imgs), test_first_imgs[0].shape[0], test_first_imgs[0].shape[1], 1) / 255
# x_test_second_img = test_second_imgs.reshape(len(test_second_imgs), test_second_imgs[0].shape[0], test_second_imgs[0].shape[1], 1) / 255


# print(x_train_first_img[0].shape)
# print(x_train_first_img[0].min(), x_train_first_img[0].max())