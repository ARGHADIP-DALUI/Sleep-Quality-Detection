import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import winsound  # <--- Core Hardware Beep Module Integration
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------------------------------
# PARAMETERS & UI CONFIGURATIONS
# ----------------------------------------------------
MAX_SEQUENCE_LENGTH = 100  # 100 frames sequence (approx. 4-5 seconds buffer)
EAR_THRESHOLD = 0.23       # Threshold for closed eyes
MAR_THRESHOLD = 0.45       # Threshold for yawning
ALARM_FRAME_TRIGGER = 15   # <--- Continuous frame threshold before alarm sounds (approx. 0.5-1 sec)

# MediaPipe Face Mesh Landmark Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308, 81, 178, 82, 311]

# ----------------------------------------------------
# FEATURE CALCULATION FUNCTIONS
# ----------------------------------------------------
def calculate_ear(landmarks, eye_indices):
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append(np.array([lm.x, lm.y]))
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

def calculate_mar(landmarks, mouth_indices):
    points = []
    for idx in mouth_indices:
        lm = landmarks[idx]
        points.append(np.array([lm.x, lm.y]))
    v_dist = np.linalg.norm(points[0] - points[1])
    h_dist = np.linalg.norm(points[2] - points[3])
    return v_dist / h_dist

# ----------------------------------------------------
# MAIN REAL-TIME MONITORING INTERFACE
# ----------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load the Pre-Trained LSTM Brain (.h5 model weights)
    model_path = os.path.join(script_dir, "../models/drowsiness_lstm_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(script_dir, "../models/abc.h5") # Fallback placeholder
        
    if not os.path.exists(model_path):
        print(f"Error: Trained model weights file not found at models folder!")
        return
        
    model = tf.keras.models.load_model(model_path)
    print(f"SUCCESS: Deep Learning LSTM Engine Loaded from: {os.path.basename(model_path)}")

    # 2. Configure Modern MediaPipe Face Mesh Options
    task_model_path = os.path.join(script_dir, "face_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=task_model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )

    # 3. Setup Rolling Memory Queue Buffer & Alarm State Counters
    frame_buffer = deque(maxlen=MAX_SEQUENCE_LENGTH)
    drowsy_frames_counter = 0  # <--- Tracks consecutive poor/sleepy frames
    
    # Activate the primary computer webcam (0 = default internal camera)
    cap = cv2.VideoCapture(0)
    print("Webcam stream channel open. Launching UI overlay window...")

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror-flip the screen frame for a natural selfie perspective
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # 4. QR-Scanner Style Interface Calculations
            box_x1, box_y1 = int(w * 0.28), int(h * 0.18)
            box_x2, box_y2 = int(w * 0.72), int(h * 0.82)
            
            # Default State Parameters
            ui_color = (255, 255, 0)  # Cyan color for default scanning
            status_text = "SCANNING PROFILE..."
            
            # Preprocess current frame for MediaPipe tracking
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]

                # Extract live scalar numbers
                left_ear = calculate_ear(landmarks, LEFT_EYE)
                right_ear = calculate_ear(landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(landmarks, MOUTH)

                # Push metrics to our live rolling buffer
                frame_buffer.append([avg_ear, mar])

                # 5. Continuous Inference Pipeline Logic
                if len(frame_buffer) == MAX_SEQUENCE_LENGTH:
                    # Convert rolling buffer queue to standard NumPy input tensor
                    input_data = np.array(frame_buffer)
                    input_data = np.expand_dims(input_data, axis=0)  # Shape becomes (1, 100, 2)

                    # Predict probability outcome via Keras Engine
                    prediction = model.predict(input_data, verbose=0)[0][0]

                    # Smooth UI Transition depending on Model Output
                    if prediction < 0.35:
                        ui_color = (0, 255, 0)      # Safe Green BGR
                        status_text = "QUALITY: GOOD | STATUS: HIGHLY ACTIVE"
                        drowsy_frames_counter = 0   # Reset alarm counter in safe state
                    elif 0.35 <= prediction <= 0.70:
                        ui_color = (0, 255, 255)    # Warning Yellow BGR
                        status_text = "QUALITY: NORMAL | ALERT: MODERATE FATIGUE"
                        drowsy_frames_counter = 0   # Reset alarm counter in moderate state
                    else:
                        ui_color = (0, 0, 255)      # Critical Danger Red BGR
                        status_text = "QUALITY: POOR | CRITICAL: HIGH DROWSINESS"
                        drowsy_frames_counter += 1  # Increment fatigue frame sequence count
                        
                        # Trigger hardware interrupt buzzer if sequence limit is crossed
                        if drowsy_frames_counter >= ALARM_FRAME_TRIGGER:
                            # Syntax: winsound.Beep(frequency_in_Hz, duration_in_ms)
                            # 2000Hz makes a loud piercing alert tone, for 300ms duration
                            winsound.Beep(2000, 300) 
                else:
                    # Buffering State UI
                    progress = int((len(frame_buffer) / MAX_SEQUENCE_LENGTH) * 100)
                    status_text = f"ANALYZING VEHICLE ENVIRONMENT... {progress}%"
                    ui_color = (255, 191, 0)
                    drowsy_frames_counter = 0

                # Render numerical metrics clearly on screen corner
                cv2.putText(frame, f"LIVE EAR: {avg_ear:.2f}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"LIVE MAR: {mar:.2f}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                status_text = "FACE PROFILE OUT OF FRAME"
                ui_color = (0, 140, 255)
                drowsy_frames_counter = 0

            # 6. UI Overlays Rendering
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), ui_color, 2)
            
            # Corner design elements for the "QR-Scanner" look
            corner_len = 25
            cv2.line(frame, (box_x1, box_y1), (box_x1 + corner_len, box_y1), ui_color, 5)
            cv2.line(frame, (box_x1, box_y1), (box_x1, box_y1 + corner_len), ui_color, 5)
            cv2.line(frame, (box_x2, box_y2), (box_x2 - corner_len, box_y2), ui_color, 5)
            cv2.line(frame, (box_x2, box_y2), (box_x2, box_y2 - corner_len), ui_color, 5)

            # Bottom Status Bar banner block panel
            cv2.rectangle(frame, (0, h - 50), (w, h), ui_color, cv2.FILLED)
            cv2.putText(frame, status_text, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            # Display the interactive final rendering output
            cv2.imshow("Real-Time Sleep Quality Scanner Dashboard", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam device pipeline closed safely.")

if __name__ == "__main__":
    main()