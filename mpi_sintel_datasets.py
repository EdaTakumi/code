import cv2 as cv
import numpy as np
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
    train_first_imgs = []
    train_second_imgs = []

    for title in train_titles:
        train_img_paths = glob.glob(train_path + 'final/' + title + '/*.png')

        train_imgs = []

        for i in range(len(train_img_paths)):
            train_img = cv.imread(train_img_paths[i])
            train_img = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)

            train_imgs.append(train_img)
        
        train_imgs = np.array(train_imgs)
        
        for i in range(len(train_imgs)-1):

            train_first_img = train_imgs[i]
            train_first_imgs.append(train_first_img)

            train_second_img = train_imgs[i+1]
            train_second_imgs.append(train_second_img)
    

    train_first_imgs = np.array(train_first_imgs)
    train_second_imgs = np.array(train_second_imgs)

    

    test_first_imgs = []
    test_second_imgs = []

    for title in test_titles:
        test_img_paths = glob.glob(test_path + 'final/' + title + '/*.png')

        test_imgs = []

        for i in range(len(test_img_paths)):
            test_img = cv.imread(test_img_paths[i])
            test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)

            test_imgs.append(test_img)

        test_imgs = np.array(test_imgs)
        
        for i in range(len(test_imgs)-1):

            test_first_img = test_imgs[i]
            test_first_imgs.append(test_first_img)

            test_second_img = test_imgs[i+1]
            test_second_imgs.append(test_second_img)
        

    test_first_imgs = np.array(test_first_imgs)
    test_second_imgs = np.array(test_second_imgs)



    # optical flow
    train_flows = []

    train_flow_paths = glob.glob(train_path + 'flow/*/*.flo')
    for i in range(len(train_flow_paths)):
        train_flow_path = train_flow_paths[i]
        train_flow = cv.readOpticalFlow(train_flow_path)
        train_flows.append(train_flow)
    
    train_flows = np.array(train_flows)


    # test_flows = np.empty(0)

    # test_flow_paths = glob.glob(test_path + 'flow/*/*.flo')
    # for i in range(len(test_flow_paths)):
    #     test_flow_path = test_flow_paths[i]
    #     test_flow = cv.readOpticalFlow(test_flow_path)
    #     test_flows.append(test_flow)

    # test_flows = np.array(test_flows)
    

    return train_first_imgs, train_second_imgs, test_first_imgs, test_second_imgs, train_flows



if __name__ == '__main__':
    train_first_imgs, train_second_imgs, test_first_imgs, test_second_imgs, train_flows = load_data()
    cv.imshow('img', train_first_imgs[0])
    cv.waitKey(0)
