import os
import cv2
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATASET_PATH = "WasteData/train"
IMG_SIZE = (64, 64)


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, IMG_SIZE)

    # Simple color histogram (FAST)
    hist = cv2.calcHist(
        [img], [0, 1, 2],
        None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()
    return hist


X, y = [], []

classes = {
    "biodegradable": 0,
    "non_biodegradable": 1
}

print("Loading images...")

for class_name, label in classes.items():
    class_path = os.path.join(DATASET_PATH, class_name)

    for root, _, files in os.walk(class_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                features = extract_features(os.path.join(root, file))
                if features is not None:
                    X.append(features)
                    y.append(label)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model accuracy:", accuracy)

joblib.dump(model, "waste_classifier.pkl")
print("Model saved as waste_classifier.pkl")