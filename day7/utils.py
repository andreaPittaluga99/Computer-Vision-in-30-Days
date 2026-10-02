import cv2
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = os.path.join(".", "face_landmarker.task")

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.FaceLandmarkerOptions(
    base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)


def get_face_landmarks(image, draw=False):

    # OpenCV BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    results = face_landmarker.detect(mp_image)

    image_landmarks = []

    if results.face_landmarks:

        landmarks = results.face_landmarks[0]

        xs = [landmark.x for landmark in landmarks]
        ys = [landmark.y for landmark in landmarks]
        zs = [landmark.z for landmark in landmarks]

        min_x = min(xs)
        min_y = min(ys)
        min_z = min(zs)

        for x, y, z in zip(xs, ys, zs):
            image_landmarks.append(x - min_x)
            image_landmarks.append(y - min_y)
            image_landmarks.append(z - min_z)

    return image_landmarks
