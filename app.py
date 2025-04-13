from tensorflow.keras.models import load_model
import cv2
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
import os

model = load_model('emotion_detect_model.h5')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.75)

while True:
    success, img = cap.read()
    img, faces = detector.findFaces(img, draw=True)

    if faces:
        for face in faces:
            x, y, w, h = face['bbox']
            x, y = max(0, x), max(0, y)
            cropped_face = img[y:y+h, x:x+w]
            gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))  # shape for model

            # Predict
            result = model.predict(reshaped)
            emotion_idx = np.argmax(result)
            emotion = class_labels[emotion_idx]

            # Display result
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

    cv2.imshow("Emotion Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
