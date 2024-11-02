import cv2
import numpy as np
import os
import sqlite3
import time
import logging

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)

# Initialize face detection and recognition
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    logging.error("Error: Could not open camera.")
    exit()

if facedetect.empty():
    logging.error("Error: Could not load haarcascade_frontalface_default.xml")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

if not os.path.exists("recognizer/trainingdata.yml"):
    logging.error("Error: trainingdata.yml not found.")
    exit()

recognizer.read("recognizer/trainingdata.yml")

def get_profile(id):
    """Retrieve profile from the database."""
    with sqlite3.connect("sqlite.db") as conn:
        cursor = conn.execute("SELECT * FROM STUDENTS WHERE Id=?", (id,))
        profile = cursor.fetchone()  # Directly fetch the first matching row
    return profile

def draw_text(img, text, pos):
    """Helper function to draw text on the image."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

prev_frame_time = 0  # For FPS calculation

try:
    while True:
        ret, img = cam.read()
        if not ret:
            logging.warning("Failed to capture image.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 50:  # Confidence threshold
                profile = get_profile(id)
                logging.info(f"Recognized Profile: {profile}")
            else:
                profile = None
                logging.warning("Face not recognized.")

            if profile is not None:
                labels = ["Name", "Age"]  # Add more labels as needed
                for i, label in enumerate(labels):
                    draw_text(img, f"{label}: {str(profile[i + 1])}", (x, y + h + 20 + i * 25))

        # FPS Calculation
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        draw_text(img, f"FPS: {int(fps)}", (10, 50))

        # Display the image with the drawn faces and text
        cv2.imshow("FACE", img)

        # Exit on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    # Release resources properly
    cam.release()
    cv2.destroyAllWindows()
    logging.info("Resources released. Program terminated.")





