import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model = load_model("sign_language_model.keras")

# Load the MediaPipe hand detection module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=3, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Set threshold for confidence in model prediction
confidence_threshold = 0.8  # Adjust as needed

# Function to preprocess frames (resize, convert to grayscale, etc.)
def preprocess_frame(frame):
    if frame is None or frame.size == 0:  # Check if frame is empty
        return None

    # Resize the frame to 28x28 (same size as the MNIST dataset)
    resized_frame = cv2.resize(frame, (28, 28))
    if resized_frame.size == 0:  # Check if resized frame is empty
        return None

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values to range [0, 1]
    normalized_frame = gray_frame / 255.0
    # Reshape the frame to match the input shape of the model
    processed_frame = np.expand_dims(normalized_frame, axis=0)
    processed_frame = np.expand_dims(processed_frame, axis=-1)
    return processed_frame

# Function to capture frames from the camera
def capture_frames():
    cap = cv2.VideoCapture(0)  # Use the default camera (you may need to change the index if you have multiple cameras)
    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            print("Failed to capture frame")
            break
        
        # Detect hands in the frame
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Convert landmarks to list of tuples (x, y)
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]
                # Draw bounding box around hand if landmarks are available
                bbox = cv2.boundingRect(np.array(landmarks, dtype=np.float32))
                x, y, w, h = bbox
                
                # Expand the bounding box
                expansion_factor = 1.5  # Adjust as needed
                new_w = int(w * expansion_factor)
                new_h = int(h * expansion_factor)
                new_x = max(0, x - (new_w - w) // 2)
                new_y = max(0, y - (new_h - h) // 2)
                
                cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
                
                # Preprocess the hand region for classification
                hand_region = preprocess_frame(frame[new_y:new_y+new_h, new_x:new_x+new_w])
                if hand_region is not None:  # Check if preprocessing succeeded
                    # Make predictions
                    prediction = model.predict(hand_region)
                    # Convert prediction to label and confidence
                    predicted_label = np.argmax(prediction)
                    confidence = np.max(prediction)
                    # If confidence is above threshold, display translation
                    if confidence > confidence_threshold:
                        # Add title showing translation (convert label number to letter)
                        letter = chr(ord('A') + predicted_label)
                        translation_text = f"Translation: {letter}"
                        cv2.putText(frame, translation_text, (new_x, new_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cap.release()
    cv2.destroyAllWindows()


# Call the function to capture frames from the camera and make predictions
capture_frames()
