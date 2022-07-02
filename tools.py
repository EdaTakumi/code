import cv2 as cv
import matplotlib
import numpy as np
import glob
import matplotlib.pyplot as plt


# 静止画から動画像を作成
def MakeMovie(size, input_path, output_path, fps=5):
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

# floファイルの可視化（カラー）
def my_vis(flow):
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype = np.uint8)
    hsv[..., 0] = ang / (2 * np.pi) * 180
    hsv[..., 1] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    hsv[..., 2] = 255
    my_flow_color = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    my_flow_color = cv.cvtColor(my_flow_color, cv.COLOR_BGR2RGB)
    return my_flow_color


if __name__ == '__main__':
    flow = cv.readOpticalFlow("C:\\dataset\\scene-flow\\MPI-Sintel\\MPI-Sintel-complete\\training\\flow\\alley_1\\frame_0001.flo")
    print(flow)
    my_flow_color = my_vis(flow)
    cv.imshow('flo', my_flow_color)
    cv.waitKey(0)
