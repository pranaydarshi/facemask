import pygame
import emoji
from keras.models import load_model
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
threshold = 0.90
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX
model = load_model("MyTrainingModel.h5")


def preprocessing(img):
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def get_className(classNo, prob):
    if classNo == 0:
        return "Mask " + " " + str(round(prob * 100, 2)) + "%"

    elif classNo == 1:
        return "No Mask " + str(round(prob * 100, 2)) + "%"


while True:
    success, imgOrignal = cap.read()
    if not success:
        continue
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = imgOrignal[y : y + h, x : x + h]
        img = cv2.resize(crop_img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)

        prediction = model.predict(img)
        classIndex = np.argmax(prediction)
        probabilityValue = np.amax(prediction)

        if probabilityValue > threshold:
            if classIndex == 0:
                cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
                cv2.putText(
                    imgOrignal,
                    get_className(classIndex, probabilityValue),
                    (x, y - 10),
                    font,
                    0.75,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            elif classIndex == 1:
                cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 0, 255), -2)
                cv2.putText(
                    imgOrignal,
                    get_className(classIndex, probabilityValue),
                    (x, y - 10),
                    font,
                    0.75,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    cv2.imshow("Result", imgOrignal)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
