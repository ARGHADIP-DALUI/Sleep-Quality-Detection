import os
import cv2
import numpy as np
import urllib.request  # Built-in library to download files via URL
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Indices for Eyes and Mouth (MediaPipe specific landmark connections)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308, 81, 178, 82, 311]  # Inner lip tracking indices

def get_ear(landmarks, eye_indices):
    """Calculates Eye Aspect Ratio (EAR) using Euclidean distances."""
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append(np.array([lm.x, lm.y]))
    
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

def get_mar(landmarks, mouth_indices):
    """Calculates Mouth Aspect Ratio (MAR) using Euclidean distances."""
    points = []
    for idx in mouth_indices:
        lm = landmarks[idx]
        points.append(np.array([lm.x, lm.y]))
    
    v_dist = np.linalg.norm(points[0] - points[1])
    h_dist = np.linalg.norm(points[2] - points[3])
    return v_dist / h_dist

def process_all_videos(base_path):
    """Iterates through active/sleepy raw directories using the modern MediaPipe Tasks API."""
    categories = ['active', 'sleepy']
    
    # Get absolute path to the task model file located in the src directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "face_landmarker.task")
    
    # AUTOMATIC DOWNLOAD: If the model file is missing, Python will download it right now
    if not os.path.exists(model_path):
        print("Model file 'face_landmarker.task' is missing. Downloading it automatically...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Successfully downloaded 'face_landmarker.task'!")
        except Exception as e:
            print(f"Failed to download the model file automatically: {e}")
            return

    # Configure modern MediaPipe Face Landmarker options
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )

    # Initialize the modern Face Landmarker detector
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for category in categories:
            folder_path = os.path.join(base_path, category)
            if not os.path.exists(folder_path):
                print(f"Directory not found: {folder_path}")
                continue

            print(f"--- Processing {category} videos ---")
            
            for video_name in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video_name)
                cap = cv2.VideoCapture(video_path)
                video_data = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR video frames to RGB for MediaPipe inference
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # MediaPipe modern API requires wrapping the frame into an Image Object
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    
                    # Run face inference
                    detection_result = landmarker.detect(mp_image)

                    if detection_result.face_landmarks:
                        # Extract the 468/478 coordinate objects list
                        landmarks = detection_result.face_landmarks[0]
                        
                        left_ear = get_ear(landmarks, LEFT_EYE)
                        right_ear = get_ear(landmarks, RIGHT_EYE)
                        avg_ear = (left_ear + right_ear) / 2.0
                        mar = get_mar(landmarks, MOUTH)
                        
                        video_data.append([avg_ear, mar])

                cap.release()
                
                # Formulate directory paths and save the extracted data structures
                output_filename = video_name.split('.')[0] + ".npy"
                output_dir = f"data/processed/{category}/"
                os.makedirs(output_dir, exist_ok=True)
                
                np.save(os.path.join(output_dir, output_filename), np.array(video_data))
                print(f"Successfully processed and saved: {output_filename}")

if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw/"
    process_all_videos(RAW_DATA_PATH)