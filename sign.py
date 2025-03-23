import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the trained model
model = tf.keras.models.load_model('sign_language_model.keras')

# Map indices to labels (adjust based on your dataset)
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
    # Note: J and Z are missing as they involve motion
}

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def extract_hand_region(frame, hand_landmarks):
    """Extract hand region from the frame using landmarks"""
    h, w, _ = frame.shape
    
    # Get bounding box of hand
    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
    
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    # Extract hand region
    hand_region = frame[y_min:y_max, x_min:x_max]
    
    return hand_region, (x_min, y_min, x_max, y_max)

def preprocess_for_prediction(hand_img):
    """Preprocess hand image for model prediction"""
    # Convert to grayscale
    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28))
    
    # Normalize
    normalized = resized / 255.0
    
    # Reshape for model input
    input_data = normalized.reshape(1, 28, 28, 1)
    
    return input_data

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture video")
        break
    
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract and preprocess hand region
            hand_img, (x_min, y_min, x_max, y_max) = extract_hand_region(frame, hand_landmarks)
            
            if hand_img.size > 0:  # Check if hand region is valid
                # Preprocess for prediction
                input_data = preprocess_for_prediction(hand_img)
                
                # Make prediction
                prediction = model.predict(input_data, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                
                # Get the predicted letter
                if predicted_class in label_map and confidence > 0.7:
                    letter = label_map[predicted_class]
                    
                    # Display the prediction
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"{letter} ({confidence:.2f})", 
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Sign Language Recognition', frame)
    
    # Exit on ESC key
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()