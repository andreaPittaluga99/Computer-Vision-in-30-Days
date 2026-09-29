from img2vec_pytorch import Img2Vec
from PIL import Image
import os
import pickle

img2vec = Img2Vec()

img_path = os.path.join(".", "data", "weather_dataset", "val", "cloudy", "cloudy2.jpg")

img = Image.open(img_path)

features = img2vec.get_vec(img)

with open(os.path.join(".", "model.p"), "rb") as f:
    model = pickle.load(f)

prediction = model.predict([features])

print(prediction)
