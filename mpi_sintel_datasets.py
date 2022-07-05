from PIL import Image
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import os
import glob


def load_data():
    # input path
    train_path = 'C:/datasets/MPI-Sintel/MPI-Sintel-complete/training/'

    train_titles = ['alley_1', 'alley_2', 'ambush_2', 'ambush_4', 'ambush_5', 'ambush_6', 'ambush_7',
                      'bamboo_1', 'bamboo_2', 'bandage_1', 'bandage_2', 'cave_2', 'cave_4',
                      'market_2', 'market_5', 'market_6', 'mountain_1', 'shaman_2', 'shaman_3',
                      'sleeping_1', 'sleeping_2', 'temple_2', 'temple_3']


    test_path = 'C:/datasets/MPI-Sintel/MPI-Sintel-complete/test/'

    test_titles = ['ambush_1', 'ambush_3', 'bamboo_3', 'cave_3', 'market_1', 'market_4', 'mountain_2',
                   'PERTURBED_market_3', 'PERTURBED_shaman_1', 'temple_1', 'tiger', 'wall']



    # image pairs
    train_pairs = []

    for title in train_titles:
        train_img_paths = glob.glob(train_path + 'final/' + title + '/*.png')
        for i in range(len(train_img_paths)-1):
            train_first_img_path = train_img_paths[i]
            train_second_img_path = train_img_paths[i+1]

            train_pair = []

            train_first_img = cv.imread(train_first_img_path)
            train_pair.append(train_first_img)

            train_second_img = cv.imread(train_second_img_path)
            train_pair.append(train_second_img)

            train_pairs.append(train_pair)

    

    test_pairs = []

    for title in test_titles:
        test_img_paths = glob.glob(test_path + 'final/' + title + '/*.png')
        for i in range(len(test_img_paths)-1):
            test_first_img_path = test_img_paths[i]
            test_second_img_path = test_img_paths[i+1]

            test_pair = []

            test_first_img = cv.imread(test_first_img_path)
            test_pair.append(test_first_img)

            test_second_img = cv.imread(test_second_img_path)
            test_pair.append(test_second_img)

            test_pairs.append(test_pair)



    # optical flow
    train_flows = []

    train_flow_paths = glob.glob(train_path + 'flow/*/*.flo')
    for i in range(len(train_flow_paths)):
        train_flow_path = train_flow_paths[i]
        train_flow = cv.readOpticalFlow(train_flow_path)
        train_flows.append(train_flow)


    # test_flows = []

    # test_flow_paths = glob.glob(test_path + 'flow/*/*.flo')
    # for i in range(len(test_flow_paths)):
    #     test_flow_path = test_flow_paths[i]
    #     test_flow = cv.readOpticalFlow(test_flow_path)
    #     test_flows.append(test_flow)
    

    # return train_pairs, test_pairs, train_flows, test_flows
    return train_pairs, test_pairs, train_flows



if __name__ == '__main__':
    x_train, x_test, t_train = load_data()
