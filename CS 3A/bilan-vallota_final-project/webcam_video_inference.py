import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import pandas as pd
from collections import deque
import os

# 1. Define the Model Class (Must match the training architecture exactly)
class LandmarkLSTM(nn.Module):
    def __init__(self, input_size=126, hidden_size=512, num_layers=2, num_classes=105):
        super(LandmarkLSTM, self).__init__()
        
        # Input embedding layer to learn better representations
        self.embedding = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        
        # Bidirectional LSTM for temporal context
        self.lstm = nn.LSTM(
            256, hidden_size, num_layers, 
            batch_first=True, 
            dropout=0.3, 
            bidirectional=True
        )
        
        # Attention mechanism to focus on important frames
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, features = x.size()
        
        # Embed each frame
        x = x.reshape(batch_size * seq_len, features)
        x = self.embedding(x)
        x = x.reshape(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention: Learn which frames are most important
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Classify
        out = self.fc(attended)
        return out

def load_labels(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # Create a dictionary mapping ID to Label
        return dict(zip(df['id'], df['label']))
    except Exception as e:
        print(f"Error loading labels: {e}")
        return None

def main():
    # --- Configuration ---
    MODEL_PATH = 'fsl_video_model_best.pth'
    LABELS_PATH = r"FSL-105 A dataset for recognizing 105 Filipino sign language videos\FSL-105 A dataset for recognizing 105 Filipino sign language videos\labels.csv"
    SEQUENCE_LENGTH = 120  # Must match max_frames from training
    INPUT_SIZE = 126       # 2 hands * 21 landmarks * 3 coords
    NUM_CLASSES = 105      # Number of classes in FSL-105
    CONFIDENCE_THRESHOLD = 0.7 # Only show prediction if confidence is high

    # --- Load Resources ---
    print("Loading labels...")
    id_to_label = load_labels(LABELS_PATH)
    if id_to_label is None:
        return

    print("\nAvailable Classes:")
    for idx, label in sorted(id_to_label.items()):
        print(f"{idx}: {label}")
    print("\n")

    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LandmarkLSTM(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Initialize MediaPipe ---
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # --- Webcam Setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # State variables
    recorded_frames = []
    is_recording = False
    prediction_text = "Press SPACE to record"
    confidence_text = ""
    
    print(f"Starting inference. Press 'SPACE' to start/stop recording. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for natural viewing
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Extract Landmarks with normalization
        frame_data = np.zeros(INPUT_SIZE, dtype=np.float32)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get handedness label
                label = results.multi_handedness[idx].classification[0].label
                
                # Flatten landmarks (x, y, z)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                landmarks = np.array(landmarks, dtype=np.float32)
                
                # CRITICAL: Normalize relative to wrist (landmark 0)
                wrist_x, wrist_y, wrist_z = landmarks[0], landmarks[1], landmarks[2]
                for i in range(0, len(landmarks), 3):
                    landmarks[i] -= wrist_x
                    landmarks[i+1] -= wrist_y
                    landmarks[i+2] -= wrist_z
                
                # Assign to specific slice based on label
                if label == 'Left':
                    frame_data[0:63] = landmarks
                else:  # Right
                    frame_data[63:126] = landmarks

        # Handle Recording
        should_predict = False
        
        if is_recording:
            recorded_frames.append(frame_data)
            # Visual indicator
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"Recording: {len(recorded_frames)}/{SEQUENCE_LENGTH}", (50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Auto-stop if full
            if len(recorded_frames) >= SEQUENCE_LENGTH:
                is_recording = False
                should_predict = True
        else:
            cv2.putText(frame, f"Result: {prediction_text} {confidence_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to start", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Key Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32: # Spacebar
            if is_recording:
                # Stop recording manually
                is_recording = False
                should_predict = True
            else:
                # Start recording
                is_recording = True
                recorded_frames = []
                prediction_text = "..."
                confidence_text = ""

        # Prediction Logic
        if should_predict and len(recorded_frames) > 0:
            print(f"Predicting on {len(recorded_frames)} frames...")
            
            # Pad with zeros if shorter than sequence length
            if len(recorded_frames) < SEQUENCE_LENGTH:
                padding = [np.zeros(INPUT_SIZE, dtype=np.float32) for _ in range(SEQUENCE_LENGTH - len(recorded_frames))]
                recorded_frames.extend(padding)
            
            # Prepare input
            input_sequence = np.array(recorded_frames, dtype=np.float32)
            input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                confidence_val = confidence.item()
                predicted_idx_val = predicted_idx.item()
                
                prediction_text = id_to_label.get(predicted_idx_val, "Unknown")
                confidence_text = f"({confidence_val:.2f})"
                print(f"Result: {prediction_text} {confidence_text}")

        cv2.imshow('FSL Video Inference', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
