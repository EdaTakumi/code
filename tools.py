import cv2 as cv
import matplotlib
import numpy as np
import glob
import matplotlib.pyplot as plt


# make movie from images
def makeMovie(size, input_path, output_path, fps=5):
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

# visualization of flow (color)
def my_vis(flow):
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang / (2 * np.pi) * 180
    hsv[..., 1] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    hsv[..., 2] = 255
    my_flow_color = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    my_flow_color = cv.cvtColor(my_flow_color, cv.COLOR_BGR2RGB)

    return my_flow_color


# visualization of flow (vector)
def drawVector(flow, figsize, per_pixel=20):
    height, width = flow.shape[0], flow.shape[1]
    plt.figure(figsize = figsize)

    for y in range(0, height, per_pixel):
        for x in range(0, width, per_pixel):
            u, v = flow[y][x]
            plt.quiver(x, height-1-y, 5*u, 5*v, width=0.003, headwidth=3, headlength=5, color='purple', angles='xy', scale_units='xy', scale=1.0)
    
    plt.xlim([0, width])
    plt.ylim([0, height])
    plt.show()


# make coordinateMat
def coordinateMat(size):
    coord_x = np.zeros(size, dtype=np.float32)
    coord_y = np.zeros(size, dtype=np.float32)
    for y in range(0, size[0], 1):
        for x in range(0, size[1], 1):
            coord_x[y][x] = x
            coord_y[y][x] = y
            
    return coord_x, coord_y


if __name__ == '__main__':
    flow = cv.readOpticalFlow("C:/datasets/MPI-Sintel/MPI-Sintel-complete/training/flow/alley_1/frame_0001.flo")
    drawVector(flow, (15.36, 6.54))
