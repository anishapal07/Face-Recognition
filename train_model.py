import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'

# Auto-create trainer directory
if not os.path.exists('trainer'):
    os.makedirs('trainer')
    print("[INFO] 'trainer' directory created.")

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    # Get paths to all images in the dataset folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]     
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        # Convert image to grayscale using Pillow, then to a numpy array
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        # Extract the user ID from the image file name
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        # Detect the face in the image to ensure we're training on the face itself
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

print("\n[INFO] Training faces. It will take a few seconds. Wait...")
faces, ids = getImagesAndLabels(path)

# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') 

print(f"\n[INFO] Model trained successfully! {len(np.unique(ids))} face(s) trained.")