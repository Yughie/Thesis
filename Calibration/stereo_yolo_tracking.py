import torch
import cv2 as cv
import numpy as np
import pickle
from ultralytics import YOLO

# Load calibration parameters
with open('stereo_calibration_data.pkl', 'rb') as f:
    calibration_data = pickle.load(f)

cameraMatrixL = calibration_data['cameraMatrixL']
distL = calibration_data['distL']
cameraMatrixR = calibration_data['cameraMatrixR']
distR = calibration_data['distR']
R = calibration_data['rot']
T = calibration_data['trans']

# Load YOLOv11 model (ensure you have ultralytics installed: `pip install ultralytics`)
model = YOLO("model/my_model.pt") # Update with your model path
model.conf = 0.5  # Confidence threshold

# Camera parameters
BASELINE = 30.0  # cm
FOCAL_LENGTH = calibration_data['cameraMatrixL'][0, 0]   # Load from calibration

# Start video capture
capL = cv.VideoCapture(0)  # Left camera (update index if needed)
capR = cv.VideoCapture(2)  # Right camera

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Failed to grab frames.")
        break

    # Undistort images
    frameL = cv.undistort(frameL, cameraMatrixL, distL)
    frameR = cv.undistort(frameR, cameraMatrixR, distR)

    # Run YOLOv11 detection
    resultsL = model(frameL)
    resultsR = model(frameR)
    
    # Extract ball coordinates
    ballL, ballR = None, None
    for result in resultsL[0].boxes.data:  # Adjusted for correct format
        class_id = int(result[5])  # Assuming class 0 is the ball
        if class_id == 0:
            x1, y1, x2, y2 = map(int, result[:4])
            ballL = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            break

    for result in resultsR[0].boxes.data:  # Adjusted for right camera
        class_id = int(result[5])
        if class_id == 0:
            x1, y1, x2, y2 = map(int, result[:4])
            ballR = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            break
    
    # Calculate 3D position if detections exist
    if ballL and ballR:
        disparity = abs(ballL[0] - ballR[0])
        if disparity > 0:
            Z = (FOCAL_LENGTH * BASELINE) / disparity
            X = ((ballL[0] - frameL.shape[1] / 2) * Z) / FOCAL_LENGTH
            Y = ((ballL[1] - frameL.shape[0] / 2) * Z) / FOCAL_LENGTH
            print(f"3D Coordinates: X={X:.2f} cm, Y={Y:.2f} cm, Z={Z:.2f} cm")
    
    # Display frames
    cv.imshow("Left Camera", frameL)
    cv.imshow("Right Camera", frameR)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv.destroyAllWindows()
