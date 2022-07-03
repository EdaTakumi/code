from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import os
import glob
import cv2 as cv

def load_data():
    trainig_path = 'C:/datasets/MPI-Sintel/MPI-Sintel-complete/training/final'
    