import os
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# preparing the data
input_directory = os.path.join(".", "clf-data")
categories = ["empty", "not_empty"]

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_directory, category)):
        img_path = os.path.join(input_directory, category, file)
        img = imread(img_path)
        img = np.asanyarray(resize(img, (15, 15)))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split

# we take 20% for testing
# with shuffle we literally shuffle the data to avoid bias while reading/labeling data
# stratify=labels preserves the original class proportions in both the training
# and test sets.
# this is especially important when the dataset is unbalanced,
# so that both sets have a similar distribution of classes
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)


# train classifier
classifier = SVC()

# map w 2 keys
# we train as many classifiers as there are possible conbination of C and gamma
parameters = [{"gamma": [0.01, 0.001, 0.0001], "C": [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance

# with that we get the best classifier that was trained
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print("{}% of samples were correcly classify".format(str(score * 100)))

# wb stands for write binaries
pickle.dump(best_estimator, open('./model.p', 'wb'))
