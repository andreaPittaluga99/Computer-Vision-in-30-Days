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

    if image is None or image.size == 0:
        return []

    height, width = image.shape[:2]

    # scale up image
    min_side = min(height, width)

    if min_side < 512:
        scale = 512 / min_side

        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )

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

        # we use nose as a reference to not have dimension problems
        nose = landmarks[164]

        nose_x = nose.x
        nose_y = nose.y
        nose_z = nose.z

        for x, y, z in zip(xs, ys, zs):
        
            x = x - nose_x
            y = y - nose_y
            z = z - nose_z

            image_landmarks.append(x)
            image_landmarks.append(y)
            image_landmarks.append(z)

    return image_landmarks
