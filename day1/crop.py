import os
import cv2

img = cv2.imread(os.path.join('.', 'data', 'bird.jpg'))

print(img.shape)

#images are just numpy arrays
cropped_img = img[200:500, 100:300]

cv2.imshow('crop', cropped_img)
cv2.waitKey(0)