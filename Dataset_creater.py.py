import sqlite3
import cv2
import numpy as np
import os

# Constants
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'
DATASET_PATH = 'dataset/'
MAX_SAMPLES = 35

# Load the Haar cascade for face detection
face_detect = cv2.CascadeClassifier(HAARCASCADE_PATH)
cam = cv2.VideoCapture(0)

# Check if the webcam and the Haar cascade were loaded successfully
if not cam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

if face_detect.empty():
    print(f"Error: Could not load Haar cascade from {HAARCASCADE_PATH}.")
    exit()

def ensure_dataset_folder_exists():
    """Create the dataset folder if it doesn't already exist."""
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

def insert_or_update(student_id, name, age):
    """Insert or update student information based on their ID."""
    with sqlite3.connect("database.db") as conn:
        cursor = conn.execute("SELECT * FROM STUDENTS WHERE ID=?", (student_id,))
        is_record_exist = cursor.fetchone() is not None  # Check if a record exists

        if is_record_exist:
            conn.execute("UPDATE STUDENTS SET Name=?, Age=? WHERE ID=?", (name, age, student_id))
            print(f"Updated student {name}'s record.")
        else:
            conn.execute("INSERT INTO STUDENTS (ID, Name, Age) VALUES (?, ?, ?)",
                         (student_id, name, age))
            print(f"Inserted new student record for {name}.")
        conn.commit()

def capture_faces(student_id):
    """Capture face samples for a given student."""
    sample_num = 0
    ensure_dataset_folder_exists()

    while sample_num < MAX_SAMPLES:
        ret, img = cam.read()
        if not ret:
            print("Failed to capture image from webcam.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            sample_num += 1
            # Save the face image
            cv2.imwrite(f"{DATASET_PATH}/user.{student_id}.{sample_num}.jpg", gray[y:y+h, x:x+w])
            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the current frame with detected faces
        cv2.imshow('Img', img)

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested to exit.")
            break

    print(f"Collected {sample_num} samples for ID {student_id}.")

# Get user input with basic validation
while True:
    try:
        student_id = int(input("Enter your ID: "))
        name = input("Enter your name: ").strip()
        age = int(input("Enter your age: "))
        if name and age > 0:
            break
        else:
            print("Invalid input. Please enter valid name and age.")
    except ValueError:
        print("Invalid input. Please enter numeric values for ID and age.")

# Insert or update the student record
insert_or_update(student_id, name, age)

# Start capturing face samples
capture_faces(student_id)

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
print("Camera released and all windows closed.")


