import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Define the path to your hand landmark model.
model_path = '/media/ponsankar/Hold/PROCV_01/hand_landmarker.task'

# Calibration factor: Replace with your own factor based on setup calibration.
# This value is distance in cm per pixel. You can calculate it by measuring known distances in your setup.
calibration_factor = 0.026  # Example: 1 pixel = 0.026 cm

# MediaPipe Task setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to process results and draw landmarks
def process_result(frame, result: HandLandmarkerResult):
    if result.hand_landmarks:
        h, w, _ = frame.shape  # Frame dimensions
        touch_points = []  # Store touch points (thumb tips) for both hands

        for hand_idx, hand_landmark in enumerate(result.hand_landmarks):
            # Extract thumb tip (landmark 4) and index finger tip (landmark 8)
            thumb_tip = hand_landmark[4]
            index_tip = hand_landmark[8]

            # Convert to pixel coordinates
            thumb_point = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_point = (int(index_tip.x * w), int(index_tip.y * h))

            # Calculate distance between thumb tip and index finger tip
            distance = calculate_distance(thumb_point, index_point)

            # Check if thumb and index finger are touching
            if distance < 20:  # Threshold for "touching"
                touch_points.append(thumb_point)  # Add thumb point to touch points
                cv2.circle(frame, thumb_point, 10, (0, 0, 255), -1)  # Mark the touching point
                cv2.putText(frame, f"Touch!", (thumb_point[0] + 10, thumb_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw landmarks and connections for the hand
            for idx, landmark in enumerate(hand_landmark):
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            connections = [(0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                           (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                           (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                           (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                           (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                           (0, 17)]
            for start_idx, end_idx in connections:
                start = hand_landmark[start_idx]
                end = hand_landmark[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

        # Display the distance and draw a line if two touch points are detected
        if len(touch_points) == 2:
            pixel_distance = calculate_distance(touch_points[0], touch_points[1])
            cm_distance = pixel_distance * calibration_factor  # Convert to cm
            midpoint = ((touch_points[0][0] + touch_points[1][0]) // 2,
                        (touch_points[0][1] + touch_points[1][1]) // 2)
            
            # Draw a line connecting the two touch points
            cv2.line(frame, touch_points[0], touch_points[1], (0, 255, 255), 2)
            
            # Display the distance along the line
            cv2.putText(frame, f"Distance: {cm_distance:.2f} cm", midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# OpenCV video capture setup
cap = cv2.VideoCapture(0)  # 0 for the default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create hand landmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

# Create the hand landmarker instance
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Convert the frame to RGB (MediaPipe uses RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image object from the frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect landmarks synchronously
        result = landmarker.detect(mp_image)

        # Process and draw results
        process_result(frame, result)

        # Display the frame in a window
        cv2.imshow('Hand Gesture Recognition', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
