import os
import cv2

img = cv2.imread(os.path.join('.', 'data', 'bird.jpg'))

# by default cv2 uses bgr colorspace
cv2.imshow('img', img)

# to convert we use cv2.cvtColor(...)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# colors looks different because of the convertion
# here the value of B becomes value of R etc...

# we have successfully rearranged the channel values, but imshow aspects an image with the bgr format, 
# so the new red channel is interpreted as blue and the new blue channel is interpreted as red  
cv2.imshow('img_rgb', img_rgb)


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img_gray', img_gray)

#hsv stands for hue, saturation, value
# for example if we want to detect color we will probably use hsv color space
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('img_hsv', img_hsv)

cv2.waitKey(0)