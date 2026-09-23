import cv2

# argument is the number of camera we want to capture
cam = cv2.VideoCapture(0)

# we dont use ret = true since we always have new frames to read
while True:
    ret, frame = cam.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
