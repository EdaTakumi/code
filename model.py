import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

import tensorflow as tf
from tensorflow.keras import models, layers

import mpi_sintel_datasets as msd


train_first_imgs, train_second_imgs, test_first_imgs, test_second_imgs, train_flows = msd.load_data()

# print(len(train_first_imgs))
# print(len(train_second_imgs))
# print(len(test_first_imgs))
# print(len(test_second_imgs))
# print(len(train_flows))

# print(train_first_imgs.shape)
# print(test_first_imgs.shape)

# cv.imshow('img', train_first_imgs[0])
# cv.waitKey(0)

x_train_first = train_first_imgs.reshape(1041, 436, 1024, 1) / 255.0
x_train_second = train_second_imgs.reshape(1041, 436, 1024, 1) / 255.0

x_test_first = test_first_imgs.reshape(552, 436, 1024, 1) / 255.0
x_test_second = test_second_imgs.reshape(552, 436, 1024, 1) / 255.0


# print(x_train_first.shape)
# print(x_train_second.shape)

# print(x_train_first.min(), x_train_first.max())
# print(x_train_second.min(), x_train_second.max())

def reset_seed(seed=0):

    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed) #　random関数のシードを固定
    np.random.seed(seed) #numpyのシードを固定
    tf.random.set_seed(seed) #tensorflowのシードを固定


# シードの固定
reset_seed(0)

# モデルの構築
model = models.Sequential([
    
])
