import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data_file = "data.txt"
data = np.loadtxt(data_file)

features = data[:, :-1]
labels = data[:, -1]  # labels are only in the last column

# split the data
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, shuffle=True, stratify=labels
)

rf_classifier = RandomForestClassifier()

rf_classifier.fit(features_train, labels_train)

label_pred = rf_classifier.predict(features_test)

accuracy = accuracy_score(labels_test, label_pred)

print(f"accuracy: {accuracy * 100}")

pickle.dump(rf_classifier, open(os.path.join(".", "model.p"), "wb"))

