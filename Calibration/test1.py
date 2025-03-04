import torch
import cv2 as cv
import numpy as np
import pickle
import threading
import time
from ultralytics import YOLO

# Load calibration parameters
with open('stereo_calibration_data.pkl', 'rb') as f:
    calibration_data = pickle.load(f)

cameraMatrixL = calibration_data['cameraMatrixL']
distL = calibration_data['distL']
cameraMatrixR = calibration_data['cameraMatrixR']
distR = calibration_data['distR']

# Load YOLOv11 model
model = YOLO("model/my_model.pt")  # Update with your model path
model.conf = 0.25  # Lower confidence threshold

# Camera parameters
BASELINE = 30.0  # cm
FOCAL_LENGTH = cameraMatrixL[0, 0]

# Global storage
frame_dict = {}
processed_frames = {}  # New: Store processed frames separately
lock = threading.Lock()
fps_dict = {0: 0, 2: 0}  # FPS tracking
process_fps = 0  # Processing FPS

def test(camera_id):
    cap = cv.VideoCapture(camera_id)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_FPS, 120)
    
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame from camera {camera_id}")
            break
        
        frame = cv.undistort(frame, cameraMatrixL if camera_id == 0 else cameraMatrixR, 
                             distL if camera_id == 0 else distR)
        
        # FPS Calculation
        curr_time = time.time()
        fps_dict[camera_id] = int(1 / (curr_time - prev_time))
        prev_time = curr_time
        
        with lock:
            frame_dict[camera_id] = frame.copy()  # Store undistorted frames
        
    cap.release()

def process_frames():
    global process_fps
    prev_time = time.time()

    while True:
        with lock:
            if 0 in frame_dict and 2 in frame_dict:
                frameL = frame_dict[0].copy()
                frameR = frame_dict[2].copy()
            else:
                continue

        resultsL = model(frameL)
        resultsR = model(frameR)

        ballL, ballR = None, None
        ballL_box, ballR_box = None, None

        # Process Left Camera detections
        for result in resultsL:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                print(f"[Left Camera] Class={class_id}, Conf={conf:.2f}, Box=({x1},{y1}) to ({x2},{y2})")

                if class_id == 0:  # If detected object is the ball
                    ballL = ((x1 + x2) // 2, (y1 + y2) // 2)
                    ballL_box = (x1, y1, x2, y2)
                    cv.rectangle(frameL, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                    break

        # Process Right Camera detections
        for result in resultsR:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                print(f"[Right Camera] Class={class_id}, Conf={conf:.2f}, Box=({x1},{y1}) to ({x2},{y2})")

                if class_id == 0:
                    ballR = ((x1 + x2) // 2, (y1 + y2) // 2)
                    ballR_box = (x1, y1, x2, y2)
                    cv.rectangle(frameR, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break

        # Calculate 3D Coordinates
        if ballL and ballR:
            disparity = abs(ballL[0] - ballR[0])
            if disparity > 0:
                Z = (FOCAL_LENGTH * BASELINE) / disparity
                X = ((ballL[0] - frameL.shape[1] / 2) * Z) / FOCAL_LENGTH
                Y = ((ballL[1] - frameL.shape[0] / 2) * Z) / FOCAL_LENGTH

                print(f"3D Coordinates: X={X:.2f} cm, Y={Y:.2f} cm, Z={Z:.2f} cm")

                if ballL_box:
                    x1, y1, _, _ = ballL_box
                    cv.putText(frameL, f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}", 
                               (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if ballR_box:
                    x1, y1, _, _ = ballR_box
                    cv.putText(frameR, f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}", 
                               (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Update processed frames
        with lock:
            processed_frames[0] = frameL
            processed_frames[2] = frameR

        # FPS Calculation for processing
        curr_time = time.time()
        process_fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d():
    global X, Y, Z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    while True:
        ax.clear()
        ax.set_xlim(-50, 50)  # Adjust based on table size
        ax.set_ylim(-50, 50)
        ax.set_zlim(0, 100)

        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_zlabel("Z (cm)")
        
        with lock:
            if X is not None and Y is not None and Z is not None:
                ax.scatter(X, Y, Z, c='r', marker='o')  # Plot ball position

        plt.pause(0.01)  # Small delay for updating visualization
thread3D = threading.Thread(target=plot_3d, daemon=True)
thread3D.start() 

def main():
    threadL = threading.Thread(target=test, args=(0,), daemon=True)
    threadR = threading.Thread(target=test, args=(2,), daemon=True)
    threadProcessing = threading.Thread(target=process_frames, daemon=True)
    thread3D = threading.Thread(target=plot_3d, daemon=True)  # Start 3D visualization
    
    threadL.start()
    threadR.start()
    threadProcessing.start()
    thread3D.start() 
    
    while True:
        with lock:
            if 0 in processed_frames:
                frameL = processed_frames[0].copy()
                cv.putText(frameL, f"FPS: {fps_dict[0]}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.imshow("Left Camera", frameL)
            
            if 2 in processed_frames:
                frameR = processed_frames[2].copy()
                cv.putText(frameR, f"FPS: {fps_dict[2]}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.imshow("Right Camera", frameR)
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()




