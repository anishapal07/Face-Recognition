import cv2
import os
import pandas as pd

# 1. Auto-create dataset directory
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    print("[INFO] 'dataset' directory created.")

# 2. Setup Pandas CSV to map IDs to Names
csv_file = 'names.csv'
face_id = input('\nEnter user ID (must be an integer, e.g., 1) ==>  ')
face_name = input('Enter user name (e.g., John) ==>  ')

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    if int(face_id) in df['ID'].values:
        df.loc[df['ID'] == int(face_id), 'Name'] = face_name
    else:
        new_row = pd.DataFrame({'ID': [int(face_id)], 'Name': [face_name]})
        df = pd.concat([df, new_row], ignore_index=True)
else:
    df = pd.DataFrame({'ID': [int(face_id)], 'Name': [face_name]})

df.to_csv(csv_file, index=False)
print(f"[INFO] Name mapping saved to {csv_file}")

# 3. Start Video Capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Use OpenCV's built-in Haar cascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("\n[INFO] Look at the camera and wait. Capturing faces...")
count = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.imshow('Capturing Dataset - Press ESC to stop early', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' to exit
    if k == 27:
        break
    elif count >= 30: # Take 30 face samples and stop
        break

print("\n[INFO] Dataset collection complete. Cleanup in progress...")
cam.release()
cv2.destroyAllWindows()