#most common use case for blurring is to remove noise from images
# we can use:
#   blur()
#   gaussianBlur()
#   medianBlur()
#   bilateralFilter()

# official cv2 resource: https://docs.opencv.org/3.4.7/dd/d6a/tutorial_js_filtering.html

# personal note:
#   basically video from acerola etc, we define neighbour pixel w a kernel
#   and we average them

import os
import cv2

img = cv2.imread(os.path.join('.', 'data', 'bird.jpg'))
cv2.imshow('og', img)

kernel_size = 7

img_blur = cv2.blur(img, (kernel_size, kernel_size))
cv2.imshow('regular blur', img_blur)

img_gaussian_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 3)
cv2.imshow('gaussian blur', img_gaussian_blur)

img_median_blur = cv2.medianBlur(img, kernel_size)
cv2.imshow('median blur', img_median_blur)


# now lets try with a noisy image

img_noised = cv2.imread(os.path.join('.', 'data', 'boats-in-lagoon-with-noise.jpg'))
cv2.imshow('noised boats', img_noised)

img_noised_blur = cv2.blur(img_noised, (kernel_size, kernel_size))
cv2.imshow('noised regular blur', img_noised_blur)

img_noised_gaussian_blur = cv2.GaussianBlur(img_noised, (kernel_size, kernel_size), 3)
cv2.imshow('noised gaussian blur', img_noised_gaussian_blur)

img_noised_median_blur = cv2.medianBlur(img_noised, kernel_size)
cv2.imshow('noised median blur', img_noised_median_blur)


cv2.waitKey(0)