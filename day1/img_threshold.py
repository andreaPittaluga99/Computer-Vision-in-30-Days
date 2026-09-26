import os
import cv2

img = cv2.imread(os.path.join('.', 'data', 'peacock.jpg'))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# < 80 -> 0, > 80 -> 255
ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)


# useful for image segmentation

# if is not covered in the series, remember to take a proper look 
# at the acerola's video about difference of gaussian
cv2.imshow('img', img)
cv2.imshow('img thresh', thresh)


# we can also denoise using blur before applying threshold
img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 3)

# ret is the threshold value actually used. with a manually specified threshold, ret is simply that value(here is 80)
# some thresholding methods can calculate the optimal threshold automatically, such as Otsu's method:
#
#   ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
# in this case, OpenCV calculates the threshold automatically
# here ret might be 117.0, for example

ret_blur, thresh_blur = cv2.threshold(img_blurred, 80, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh blur', thresh_blur)

# check adaptive threshold
# cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
# blocksize needs to be odd and greater than 1
# constant C is subtracted from the calculated local mean
# so basically we divide an img in section and opencv calculates the optimal threshold for each region
thresh_adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 20)
cv2.imshow('adaptive', thresh_adaptive)

cv2.waitKey(0)