import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Indices for Eyes and Mouth (MediaPipe specific)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308, 81, 178, 82, 311] # Inner lip indices

def get_ear(landmarks, eye_indices):
    # Extract coordinates
    points = []
    for idx in eye_indices:
        lm = landmarks.landmark[idx]
        points.append(np.array([lm.x, lm.y]))
    
    # Calculate vertical distances
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    # Calculate horizontal distance
    C = np.linalg.norm(points[0] - points[3])
    
    return (A + B) / (2.0 * C)

def get_mar(landmarks, mouth_indices):
    points = []
    for idx in mouth_indices:
        lm = landmarks.landmark[idx]
        points.append(np.array([lm.x, lm.y]))
    
    # Simple MAR calculation: Vertical / Horizontal
    v_dist = np.linalg.norm(points[0] - points[1]) # Top and Bottom lip
    h_dist = np.linalg.norm(points[2] - points[3]) # Left and Right corner
    return v_dist / h_dist

def process_all_videos(base_path):
    categories = ['active', 'sleepy']
    
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
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    
                    left_ear = get_ear(landmarks, LEFT_EYE)
                    right_ear = get_ear(landmarks, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0
                    mar = get_mar(landmarks, MOUTH)
                    
                    video_data.append([avg_ear, mar])

            cap.release()
            
            # Save the numeric data as a .npy file
            output_filename = video_name.split('.')[0] + ".npy"
            output_dir = f"data/processed/{category}/"
            os.makedirs(output_dir, exist_ok=True)
            
            np.save(os.path.join(output_dir, output_filename), np.array(video_data))
            print(f"Successfully processed and saved: {output_filename}")

if __name__ == "__main__":
    # Path to your raw video dataset
    RAW_DATA_PATH = "data/raw/"
    process_all_videos(RAW_DATA_PATH) 