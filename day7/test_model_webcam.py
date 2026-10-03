import cv2
import os
import pickle
from collections import deque, Counter

from utils import get_face_landmarks

emotions = ["fear", "happy", "neutral", "sad"]

cap = cv2.VideoCapture(0)

model = pickle.load(open(os.path.join(".", "model.p"), "rb"))

predictions = deque(maxlen=15)

ret = True

while ret:
    ret, frame = cap.read()

    face_landmarks = get_face_landmarks(frame)

    if face_landmarks:
        # current frame prediction
        output = model.predict([face_landmarks])[0]

        predictions.append(int(output))

        stable_prediction = Counter(predictions).most_common(1)[0][0]
        cv2.putText(
            frame,
            emotions[stable_prediction],
            (10, frame.shape[0] - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
        )
    else:
        predictions.clear()
        cv2.putText(
            frame,
            "no face detected",
            (10, frame.shape[0] - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
        )
        


    cv2.imshow("frame", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
