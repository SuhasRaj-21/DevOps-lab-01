import cv2
import numpy as np
from PIL import Image
import os

def train_classifier():
    path = 'dataset'
    if not os.path.exists(path):
        os.makedirs(path)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]     
        faceSamples = []
        ids = []

        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

        return faceSamples, ids

    faces, ids = getImagesAndLabels(path)
    if not faces:
        return
        
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')
