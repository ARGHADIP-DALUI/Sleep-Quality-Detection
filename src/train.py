import os
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt  # New import for validation curves


# Configurable constants
DATA_DIR = "data/processed"
MAX_SEQUENCE_LENGTH = 100  # Normalizing all video inputs to 100 frames


def load_and_preprocess_data():
    X = []
    y = []
   
    categories = {'active': 0, 'sleepy': 1}
   
    for category, label in categories.items():
        category_path = os.path.join(DATA_DIR, category)
       
        if not os.path.exists(category_path):
            print(f"Warning: Folder {category_path} not found. Skipping.")
            continue
           
        for file_name in os.listdir(category_path):
            if file_name.endswith('.npy'):
                file_path = os.path.join(category_path, file_name)
               
                # Load the mathematical sequence (shape: [num_frames, 2])
                sequence = np.load(file_path)
               
                X.append(sequence)
                y.append(label)
               
    if len(X) == 0:
        raise ValueError("No .npy data found! Please run preprocess.py first.")
       
    # Sequence Padding (Post padding with zeros if sequence is short)
    X_padded = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding='post', dtype='float32')
    y = np.array(y)
   
    print(f"Total samples loaded: {len(X_padded)}")
    print(f"Shape of X after padding: {X_padded.shape}")  # Expected: (samples, 100, 2)
   
    # Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42, stratify=y)
   
    return X_train, X_test, y_train, y_test


def build_lstm_model(input_shape):
    """
    Designs the LSTM Network Layout (The Brain)
    """
    model = Sequential([
        # LSTM Layer 1: Captures temporal sequence of EAR/MAR
        LSTM(units=64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),  # Prevents overfitting
       
        # LSTM Layer 2: Deeper understanding of patterns over time
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
       
        # Dense Layers for classification logic simplify
        Dense(units=16, activation='relu'),
       
        # Output Layer: Sigmoid gives probability between 0 (Active) and 1 (Sleepy)
        Dense(units=1, activation='sigmoid')
    ])
   
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
   
    return model


if __name__ == "__main__":
    # Step 1: Load and split data in RAM
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
   
    # Step 2: Build the Network Layout
    input_shape = (X_train.shape[1], X_train.shape[2]) # (100, 2)
    model = build_lstm_model(input_shape)
   
    print("\n--- LSTM Model Structure Summary ---")
    model.summary()
   
    # Step 3: Start Real Training on RAM
    print("\nStarting Model Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,          # Loop tracker over dataset
        batch_size=4,       # Best for 36 samples
        verbose=1
    )
   
    # Step 4: Save permanently to Hard Disk!
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/drowsiness_lstm_model.h5"
    model.save(model_save_path)
    print(f"\nSUCCESS!!! Trained model permanently saved at: {model_save_path}")
   
    # Step 5: Plot Accuracy and Loss Curves (Validation Curve mapping)
    print("\nPlotting Accuracy and Loss curves...")
    os.makedirs("plots", exist_ok=True)
   
    plt.figure(figsize=(12, 5))
   
    # Plotting Accuracy trends
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
    plt.title('Model Accuracy Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.grid(True)
   
    # Plotting Loss trends
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title('Model Loss Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)
   
    # Save image file to directory
    plot_save_path = "plots/training_performance.png"
    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.close()
   
    print(f"SUCCESS!!! Performance curves saved at: {plot_save_path}")