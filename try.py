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

    # train_titles = ['alley_1', 'alley_2', 'ambush_2', 'ambush_4', 'ambush_5', 'ambush_6', 'ambush_7',
    #                   'bamboo_1', 'bamboo_2', 'bandage_1', 'bandage_2', 'cave_2', 'cave_4',
    #                   'market_2', 'market_5', 'market_6', 'mountain_1', 'shaman_2', 'shaman_3',
    #                   'sleeping_1', 'sleeping_2', 'temple_2', 'temple_3']

    train_titles = ['alley_1']


    test_path = 'C:/datasets/MPI-Sintel/MPI-Sintel-complete/test/'

    test_titles = ['ambush_1', 'ambush_3', 'bamboo_3', 'cave_3', 'market_1', 'market_4', 'mountain_2',
                   'PERTURBED_market_3', 'PERTURBED_shaman_1', 'temple_1', 'tiger', 'wall']



    # image pairs
    train_first_imgs = np.empty(0)
    # train_second_imgs = np.empty(0)

    for title in train_titles:
        train_img_paths = glob.glob(train_path + 'final/' + title + '/*.png')

        train_imgs = np.empty(0)

        for i in range(len(train_img_paths)):
            train_img = cv.imread(train_img_paths[i])
            train_img = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)

            train_imgs = np.append(train_imgs, train_img)
        
        for i in range(len(train_imgs)-1):

            train_first_img = train_imgs[i]
            train_first_imgs = np.append(train_first_imgs, train_first_img)

            # train_second_img = train_imgs[i+1]
            # train_second_imgs = np.append(train_second_imgs, train_second_img)

    

    # test_first_imgs = np.empty(0)
    # test_second_imgs = np.empty(0)

    # for title in test_titles:
    #     test_img_paths = glob.glob(test_path + 'final/' + title + '/*.png')

    #     test_imgs = np.empty(0)

    #     for i in range(len(test_img_paths)):
    #         # test_img = cv.imread(test_img_paths[i])
    #         # test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)

    #         test_imgs = np.append(test_imgs, test_img_paths[i])
        
    #     for i in range(len(test_imgs)-1):

    #         test_first_img = test_imgs[i]
    #         test_first_imgs = np.append(test_first_imgs, test_first_img)

    #         test_second_img = test_imgs[i+1]
    #         test_second_imgs = np.append(test_second_imgs, test_second_img)



    # optical flow
    # train_flows = np.empty(0)

    # train_flow_paths = glob.glob(train_path + 'flow/*/*.flo')
    # for i in range(len(train_flow_paths)):
    #     train_flow_path = train_flow_paths[i]
    #     train_flow = cv.readOpticalFlow(train_flow_path)
    #     train_flows = np.append(train_flows, train_flow)


    # test_flows = np.empty(0)

    # test_flow_paths = glob.glob(test_path + 'flow/*/*.flo')
    # for i in range(len(test_flow_paths)):
    #     test_flow_path = test_flow_paths[i]
    #     test_flow = cv.readOpticalFlow(test_flow_path)
    #     test_flows = np.append(test_flows, test_flow)
    

    # return train_first_imgs, train_second_imgs, test_first_imgs, test_second_imgs
    return train_first_imgs



if __name__ == '__main__':
    train_first_imgs = load_data()
    cv.imshow('img', train_first_imgs[0])
    cv.waitKey(0)

