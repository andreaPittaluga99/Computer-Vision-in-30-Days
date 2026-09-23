import os
import cv2

video_path = os.path.join('.', 'data', 'bird_30fps.mp4')
video = cv2.VideoCapture(video_path)

ret = True

# ret is a bool wich is true as long as there is at least one frame left in the video
while ret:
    ret, frame = video.read()

    if ret:
        cv2.imshow('frame', frame)
        #cv2.waitKey(33) waits for 33ms and checks for keyboard input
        # the last 8 bits identifies wich key was pressed
        # & 0xFF extract the last 8 bit of the value
        # then we compare with ord('q')
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

# release resources
video.release()
cv2.destroyAllWindows()