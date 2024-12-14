import threading
import cv2
from deepface import DeepFace
import os
import numpy as np
from scipy.spatial.distance import cosine  # Import cosine distance

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Path to the folder containing images
folder_path = r"image"  # Folder with training images

# List to store embeddings of reference images
reference_embeddings = []

# Load all images from the folder and calculate embeddings
def load_reference_images(folder_path):
    global reference_embeddings
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(file_path)
            if img is not None:
                embedding = DeepFace.represent(img, model_name="VGG-Face", enforce_detection=False)
                reference_embeddings.append(embedding[0]["embedding"])  # Store the embedding

# Run the function to load reference images and calculate embeddings
load_reference_images(folder_path)
print(f"Loaded {len(reference_embeddings)} reference images.")

# Function to log the result to a text file
def log_match_result(result):
    with open("match_results.txt", "a") as file:  # Open the file in append mode
        file.write(result + "\n")  # Write the result with a newline

def check_face(frame):
    global face_match

    try:
        # Use DeepFace to calculate the embedding of the current frame
        frame_embedding = DeepFace.represent(frame, model_name="VGG-Face", enforce_detection=False)

        # Compare frame's embedding with reference embeddings using cosine distance
        for reference_embedding in reference_embeddings:
            distance = cosine(frame_embedding[0]["embedding"], reference_embedding)  # Cosine distance

            # Threshold for considering a match
            if distance < 0.5:  # You can adjust this threshold
                face_match = True
                log_match_result("MATCH")  # Log the result in the text file
                return  # Exit once a match is found
        face_match = False
        log_match_result("NO MATCH")  # Log the result in the text file
    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 38 == 8:  # Process frame every 38th frame (you can adjust the number)
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        # Display match result on the frame
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
