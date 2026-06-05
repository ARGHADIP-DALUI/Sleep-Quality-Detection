# Real-Time Driver Sleep Quality & Fatigue Detection System

An advanced, decoupled two-stage Deep Learning pipeline that continuously monitors live camera stream inputs to evaluate a vehicle driver's fatigue indexes, alerting against micro-sleep scenarios in real-time.

---

## 🛠️ System Architecture Diagram Overview

The application architecture isolates high-dimensional raw pixel tracking workloads from the core sequence classification engine to achieve high computational efficiency.

1. **Stage 1 (Feature Transformation Layer):** High-resolution RGB video matrices from the webcam are intercepted by the **Google MediaPipe Tasks API**. It tracks 3D sub-pixel facial meshes and calculates normalized mathematical scalars: **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)**.
2. **Stage 2 (Temporal Sequence Classifier):** These compressed coordinates are stored into a rolling 100-frame First-In-First-Out (FIFO) queue (`collections.deque`) and fed directly into a trained **Sequential LSTM (Long Short-Term Memory)** network to predict driver state.

---

## 📦 Project Directory Layout

```text
Sleep-Quality-Detection/
│   requirements.txt       # Exact system software package dependencies
│   README.md              # Project layout and installation manual
│   struc.txt              # Folder structural integrity log
│   
├───data/
│   ├───processed/         # Pre-extracted spatial scalar matrices (.npy)
│   └───raw/               # Baseline multi-class video datasets (.mp4)
│
├───models/
│   └───drowsiness_lstm_model.h5   # Saved weights of the trained LSTM brain
│
├───plots/
│   └───training_performance.png   # Train/Validation Accuracy and Loss curve plots
│
└───src/
    ├───app.py             # Final Live UI "QR-Scanner" style Application
    ├───preprocess.py      # Feature extraction automation engine
    ├───train.py           # Deep Learning architecture compilation script
    └───face_landmarker.task  # Pre-compiled Google MediaPipe model weights