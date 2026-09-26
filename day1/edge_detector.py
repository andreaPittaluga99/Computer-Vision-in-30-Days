import os
import cv2
import numpy as np

img = cv2.imread(os.path.join('.', 'data', 'bird.jpg'))

# lower threshold, higher threshold
# it uses hysterisis thresholding

# gradient < lower threshold -> not an edge
# lower threshold < gradient < higher threshold -> weak edge
# gradient < lower threshold -> strong edge

# a little bit of trial and error to get the right value might be needed
img_canny = cv2.Canny(img, 100, 200)

cv2.imshow('img',img)
cv2.imshow('img_canny', img_canny)

# w dilate we make all the boarders thicker
img_canny_d = cv2.dilate(img_canny, np.ones((3,3), dtype = np.int8))
cv2.imshow('img_canny_d', img_canny_d)

# erode is the opposite
img_canny_e = cv2.erode(img_canny_d, np.ones((3,3), dtype = np.int8))
cv2.imshow('img_canny_e', img_canny_e)

cv2.waitKey(0)