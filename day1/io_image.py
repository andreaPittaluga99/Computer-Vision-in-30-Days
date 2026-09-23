import os

import cv2

img_path = os.path.join('.', 'data', 'bird.jpg')
img = cv2.imread(img_path)

cv2.imwrite(os.path.join('.', 'data', 'bird_out.jpg'), img)

cv2.imshow('img', img)

# the argument specifies how long the image is displayed, in milliseconds
# if set to 0, the image is displayed indefinitely
cv2.waitKey(0)