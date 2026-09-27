import cv2
import os
import mediapipe as mp

img_path = os.path.join("..", "data", "face.jpg")
model_path = os.path.join("..", "models", "blaze_face_short_range.tflite")


options = mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
)

# create the mediapipe face detector using the configured options
with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
    img = cv2.imread(img_path)
    # mediapipe uses rgb color space
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # we get an img in the mp format from a np array
    mp_img = mp.Image(mp.ImageFormat.SRGB, rgb_image)

    # run the model against the img
    result = detector.detect(mp_img)

    print(f"faces: {len(result.detections)}")
    for detection in result.detections:
        bbox = detection.bounding_box

        x = bbox.origin_x
        y = bbox.origin_y
        w = bbox.width
        h = bbox.height

        face = img[y : y + h, x : x + w]

        blurred_face = cv2.blur(face, (51, 51))

        img[y : y + h, x : x + w] = blurred_face

cv2.imshow("MediaPipe Face Detection", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
