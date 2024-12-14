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
scan_results = []  # To store match results during scanning
scan_duration = 3  # Duration for scanning (in seconds)
threshold = 0.5  # Distance threshold for face match

# Path to the folder containing images
folder_path = r"tsawer"

# List to store embeddings and corresponding labels
reference_data = []  # Format: [{'label': 'name', 'embedding': [values]}]

# Load all images from the folder and calculate embeddings
def load_reference_images(folder_path):
    global reference_data
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(file_path)
            if img is not None:
                embedding = DeepFace.represent(img, model_name="VGG-Face", enforce_detection=False)
                label = os.path.splitext(filename)[0]  # Use filename (without extension) as label
                reference_data.append({'label': label, 'embedding': embedding[0]["embedding"]})
    print(f"Loaded {len(reference_data)} reference images.")

# Run the function to load reference images and calculate embeddings
load_reference_images(folder_path)

def log_match_result(result):
    with open("match_results.txt", "a") as file:  # Open the file in append mode
        file.write(result + "\n")  # Write the result with a newline
# Function to compare a frame's embedding with reference embeddings
def check_face(frame):
    global scan_results

    try:
        # Use DeepFace to calculate the embedding of the current frame
        frame_embedding = DeepFace.represent(frame, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]

        # Initialize variables for tracking closest match
        best_match_label = "Unknown"
        best_distance = float("inf")

        # Compare frame's embedding with all reference embeddings
        for reference in reference_data:
            distance = cosine(frame_embedding, reference['embedding'])

            # Update best match if distance is within the threshold
            if distance < threshold and distance < best_distance:
                best_distance = distance
                best_match_label = reference['label']

        scan_results.append(best_match_label)  # Append the label of the best match
    except ValueError:
        scan_results.append("Unknown")

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
                # Count occurrences of each label
                match_count = {label: scan_results.count(label) for label in set(scan_results)}

                # Find the label with the highest count
                final_label = max(match_count, key=match_count.get)

                # Decide final result
                print(f"Final Result: {final_label}")
                log_match_result(f"FINAL RESULT: {final_label}")

            # Reset scanning variables
            scan_results = []
            start_time = time.time()

        # Display match result on the frame
        cv2.putText(frame, final_label if "final_label" in locals() else "Scanning...",
                    (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "final_label" in locals() else (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
