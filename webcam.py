import cv2
import numpy as np
import joblib


IMG_SIZE = (64, 64)
model = joblib.load("waste_classifier.pkl")


def extract_features_from_frame(frame):
    img = cv2.resize(frame, IMG_SIZE)

    hist = cv2.calcHist(
        [img], [0, 1, 2],
        None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    hist = cv2.normalize(hist, hist).flatten()
    return hist.reshape(1, -1)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    features = extract_features_from_frame(frame)
    prediction = model.predict(features)[0]

    label = "Biodegradable" if prediction == 0 else "Non-Biodegradable"

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Waste Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
