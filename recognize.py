import cv2
import numpy as np
import pandas as pd
import os

print("\n[INFO] Loading trained model and starting camera...")

recognizer = cv2.face.LBPHFaceRecognizer_create()
# Read the trained model
try:
    recognizer.read('trainer/trainer.yml')
except cv2.error:
    print("[ERROR] Could not find trainer.yml. Please run train_model.py first.")
    exit()

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the names dictionary using Pandas
try:
    df = pd.read_csv('names.csv')
    # Convert dataframe into a dictionary: {1: 'John', 2: 'Jane'}
    names_dict = dict(zip(df.ID, df.Name))
except FileNotFoundError:
    print("[ERROR] names.csv not found. Please run create_dataset.py first.")
    exit()

# Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Predict the face
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if confidence is less than 100 ==> "0" is a perfect match 
        if confidence < 100:
            name = names_dict.get(id, "Unknown ID")
            confidence_text = f"  {round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"
        
        # Display the name and confidence percentage
        cv2.putText(img, str(name), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)  
    
    cv2.imshow('Face Recognition - Press ESC to exit', img) 
    
    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

# Cleanup
print("\n[INFO] Exiting Program and cleaning up stuff")
cam.release()
cv2.destroyAllWindows()