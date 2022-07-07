import cv2 as cv
import numpy as np


def my_vis(flow):
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang / (2 * np.pi) * 180
    hsv[..., 1] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    hsv[..., 2] = 255
    my_flow_color = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    my_flow_color = cv.cvtColor(my_flow_color, cv.COLOR_BGR2RGB)

    return my_flow_color


if __name__ == '__main__':
    my_vis()
    