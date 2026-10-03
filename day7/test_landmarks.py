import cv2
import mediapipe as mp
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = os.path.join(".", "face_landmarker.task")

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.FaceLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)

ret = True

while ret:

    ret, frame = cap.read()

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    results = face_landmarker.detect(mp_image)

    if results.face_landmarks:

        landmarks = results.face_landmarks[0]

        height, width = frame.shape[:2]

        for i, landmark in enumerate(landmarks):

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.putText(
                frame,
                str(i),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 0),
                1,
            )

    cv2.imshow("landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
