# Real-Time Face Recognition System

This project is a real-time face recognition application built using Python, OpenCV, Pandas, and NumPy. It uses your computer's webcam to capture face data, trains a Local Binary Patterns Histograms (LBPH) model, and recognizes faces in real-time.

## 📂 Project Files

* `create_dataset.py`: Opens the webcam to capture 30 images of your face. It auto-creates a `dataset` folder to store the images and saves your Name and ID in a `names.csv` file.
* `train_model.py`: Reads the captured images from the `dataset` folder, trains the LBPH face recognizer, and saves the trained model in an auto-created `trainer` folder as `trainer.yml`.
* `recognize.py`: Opens the webcam in real-time, loads the trained model and the `names.csv` file, and draws a box around recognized faces with their name and confidence level.
* `requirements.txt`: Contains the list of Python libraries needed to run this project.

## ⚙️ Installation and Setup

**1. Clone or download the project**
Make sure all four files (`create_dataset.py`, `train_model.py`, `recognize.py`, and `requirements.txt`) are in the same folder.

**2. Install the required libraries**
Open your terminal or command prompt, navigate to the project folder, and run this command:
```bash
pip install -r requirements.txt

```

> **Note:** We use `opencv-contrib-python` instead of standard OpenCV because the standard version does not include the face recognition modules required for this project.

## 🚀 How to Run the Project

You must run the files in this exact order:

**Step 1: Create your dataset**

```bash
python create_dataset.py

```

* The terminal will ask for an **ID** (enter a number, like `1`) and a **Name** (like `John`).
* Look at your webcam. The program will take 30 pictures of your face and close automatically.

**Step 2: Train the model**

```bash
python train_model.py

```

* Wait a few seconds for the script to process your images and train the model. It will print a success message when finished.

**Step 3: Recognize faces**

```bash
python recognize.py

```

* A new window will open showing your webcam feed.
* If everything is set up correctly, you will see a green box around your face showing your name and the match percentage.
* Press the **ESC** key on your keyboard to close the window and exit the program.

## 🛑 Troubleshooting

* **Error: `cv2.face` module not found:** Make sure you installed `opencv-contrib-python` and not just `opencv-python`.
* **Error: `trainer.yml` not found:** You need to run `train_model.py` before running the recognition script.
* **Camera not opening:** Check if another application (like Zoom or Teams) is currently using your webcam.

```