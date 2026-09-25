import os
import cv2

img = cv2.imread(os.path.join('.', 'data', 'bird.jpg'))

#width, height
resized = cv2.resize(img, (400, 600))

print(img.shape)
print(resized.shape)

cv2.imshow('img', img)
cv2.imshow('resized', resized)
cv2.waitKey(0)

