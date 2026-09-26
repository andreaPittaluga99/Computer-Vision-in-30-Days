import numpy as np
import cv2

def get_limits(color):
    #basically makes an image 1x1 of this color
    c = np.uint8([[color]])
    # converts the 1x1 image in the hsv color space
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    # hsvC [0][0][0] so first row, first pixel, H channel

    # for example we get 60
    # lower_limit = 60 - 10, 100, 100
    # we get lower_limit = (50, 100, 100)

    # so basically we get +- 10 for the hue, 
    # 100 <= saturation <= 255
    # 100 <= value <= 255
    lower_limit = hsvC [0][0][0] - 10, 100, 100
    upper_limit = hsvC [0][0][0] + 10, 255, 255

    # cv2.inRange() expects numpy array, so we cast our touple to np
    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)

    return lower_limit, upper_limit