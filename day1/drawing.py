import os
import cv2

img = cv2.imread(os.path.join('.', 'data', 'whiteboard.jpg'))

# remeber shape return y/x while coord in func use x/y
print(img.shape)
# img, starting point, end point, color, thickness
cv2.line(img, (50, 60), (180,200), (0, 255, 0), 3)

#first point upper left corner, second point bottom right, color, thickness

#              X →
#        0 ─────────────────────── width
#        │
#        │  (0,0)
#        │    ●
#        │     \
#        │      \
#      Y ↓       \
#        │        \
#        │         ● (300,500)
#        │
#        │
#      height

#if we use thickness == -1 the shape will be filled
cv2.rectangle(img, (100, 60), (300,300), (255, 0, 0), 5)

#center, radious
cv2.circle(img, (200, 200), 50, (0, 0, 255), -1)

# tetx, bottom left corner of text, font of choice, size of text, color, thickness of text strokes
cv2.putText(img, 'interesting statement', (150, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

cv2.imshow('board', img)
cv2.waitKey(0)

