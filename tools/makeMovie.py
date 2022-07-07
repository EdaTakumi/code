import cv2 as cv
import numpy as np
import glob


def makeMovie(size, input_path, output_path, fps=5):

    # size -> (width, height)
    # input_path -> './*png'
    # output_path -> './~.mp4'

    pic_data = glob.glob(input_path)
    fourcc = cv.VideoWriter_fourcc('m','p','4','v')
    save = cv.VideoWriter(output_path, fourcc, fps, size)

    print('Now loading...')

    for i in range(len(pic_data)):
        img = pic_data[i]
        img = cv.imread(img)
        img = cv.resize(img, size)
        save.write(img)
    
    print('Completed!')

    save.release()


if __name__ == '__main__':
    makeMovie()
    