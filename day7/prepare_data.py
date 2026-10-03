import os
import cv2
import numpy as np
from utils import get_face_landmarks

data_dir = os.path.join(".", "data")

output = []

invalid_images = 0
valid_images = 0


for emotion_idx, emotion in enumerate(os.listdir(data_dir)):
    for image_name in os.listdir(os.path.join(data_dir, emotion)):
        img_path = os.path.join(data_dir, emotion, image_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Cant read {img_path}")
            continue

        face_landmarks = get_face_landmarks(img)

        if not face_landmarks:
            print(f"No face detected in {img_path}")
            invalid_images += 1
            continue

        face_landmarks.append(emotion_idx)

        output.append(face_landmarks)
        valid_images += 1


print(f"invalid images: {invalid_images}")
print(f"valid images  : {valid_images}")
np.savetxt("data.txt", np.asanyarray(output))
