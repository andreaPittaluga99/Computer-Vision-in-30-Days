import os
import cv2

img = cv2.imread(os.path.join('.', 'data', 'flock-of-birds.jpg'))

# < 127 -> 255, > 127 -> 0

# if we want to work w contours we need to have the designated onj in white
ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # all the small values are probably noise, so we filter them out
    if cv2.contourArea(cnt) > 200:
        # cnt is the current contour
        # -1 draw all contours contained in cnt (here cnt contains only the current contour)
        # color 
        # thickness
        # cv2.drawContours(img, cnt, -1, (0, 0, 255), 1)


        #return point + dimension of rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('img', img)
cv2.imshow('thresh', thresh)


cv2.waitKey(0)

