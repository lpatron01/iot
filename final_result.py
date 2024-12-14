import threading
import cv2
from deepface import DeepFace
import os
import numpy as np
from scipy.spatial.distance import cosine
import time

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables
counter = 0
face_match = False
scan_results = []  # To store match results during scanning
scan_duration = 3  # Duration for scanning (in seconds)

# Path to the folder containing images
folder_path = r"image"

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
                reference_embeddings.append(embedding[0]["embedding"])

# Run the function to load reference images and calculate embeddings
load_reference_images(folder_path)
print(f"Loaded {len(reference_embeddings)} reference images.")

# Function to log the result to a text file
def log_match_result(result):
    with open("match_results.txt", "a") as file:  # Open the file in append mode
        file.write(result + "\n")  # Write the result with a newline

# Function to compare a frame's embedding with reference embeddings
def check_face(frame):
    global scan_results

    try:
        # Use DeepFace to calculate the embedding of the current frame
        frame_embedding = DeepFace.represent(frame, model_name="VGG-Face", enforce_detection=False)

        # Compare frame's embedding with reference embeddings using cosine distance
        for reference_embedding in reference_embeddings:
            distance = cosine(frame_embedding[0]["embedding"], reference_embedding)

            # Threshold for considering a match
            if distance < 0.5:  # You can adjust this threshold
                scan_results.append("MATCH")
                return
        scan_results.append("NO MATCH")
    except ValueError:
        scan_results.append("NO MATCH")

start_time = time.time()

while True:
    ret, frame = cap.read()

    if ret:
        current_time = time.time()

        # Check if scanning is within the defined duration
        if current_time - start_time < scan_duration:
            if counter % 10 == 0:  # Process every 10th frame (adjustable)
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            counter += 1
        else:
            # Determine the majority result after scanning
            if scan_results:
                match_count = scan_results.count("MATCH")
                no_match_count = scan_results.count("NO MATCH")

                # Decide final result based on majority
                face_match = match_count > no_match_count
                final_result = "MATCH" if face_match else "NO MATCH"
                print(f"Final Result: {final_result}")
                log_match_result(f"FINAL RESULT: {final_result}")

            # Reset scanning variables
            scan_results = []
            start_time = time.time()

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
