import cv2
from PIL import Image
from utils import get_limits

cap = cv2.VideoCapture(0)

yellow = [0, 255, 255] # bgr color space

while True:
    ret, frame = cap.read()

    # we use hsv color space cause is easier to check for color
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_limit, upper_limit = get_limits(yellow)
    
    mask = cv2.inRange(hsv_img, lower_limit, upper_limit)
    mask_ = Image.fromarray(mask)
    # we use pillow cause is super easy to get bounding box
    bbox = mask_.getbbox()
    print(bbox)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()