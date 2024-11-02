

import os  # Can read and write files
import cv2  # Open the camera
import numpy as np  # For array operations
from PIL import Image  # Image file read and write

# Create the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"

def get_the_image_with_id(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]  # Image paths
    faces = []
    ids = []
    for single_image_path in image_paths:
        face_img = Image.open(single_image_path).convert('L')  # Convert to grayscale
        face_np = np.array(face_img, np.uint8)
        # Extract ID from the filename (assuming filename format: "user.id.something.jpg")
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        print(f"ID: {id}")

        faces.append(face_np)
        ids.append(id)

        cv2.imshow("Training on image...", face_np)
        cv2.waitKey(100)  # Correct: cv2.waitKey, not waitkey

    return np.array(ids), faces

# Train the recognizer with the dataset
ids, faces = get_the_image_with_id(path)
recognizer.train(faces, ids)  # Pass `ids`, not `id`
recognizer.save("recognizer/trainingdata.yml")
cv2.destroyAllWindows()
