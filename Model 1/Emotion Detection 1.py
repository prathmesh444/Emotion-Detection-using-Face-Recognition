import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras_preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier(r"C:\Users\hp\minor_project\haarcascade_frontalface_default.xml")
classifier = my_reloaded_model = tf.keras.models.load_model(
    r"C:\Users\hp\minor_project\Latest_model.h5",
    custom_objects={'KerasLayer': hub.KerasLayer}
)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (48, 48), 2)
        # roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(frame, (48, 48))

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            print(prediction)
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
