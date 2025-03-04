import torch
import cv2 as cv
import numpy as np
import pickle
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
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

# Camera parameters
BASELINE = 30.0  # cm
FOCAL_LENGTH = cameraMatrixL[0, 0]

# Global variables for frame storage and locks
frame_dict = {}
processed_frames = {} 
lock = threading.Lock()
fps_dict = {0: 0, 2: 0}  # Stores FPS for both cameras
process_fps = 0  # Stores FPS for YOLO processing

# List to store detected 3D ball positions
ball_positions = []

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
            frame_dict[camera_id] = frame

    cap.release()

def process_frames():
    global process_fps
    prev_time = time.time()

    while True:
        with lock:
            if 0 in frame_dict and 2 in frame_dict:  # Ensure both frames exist
                frameL = frame_dict[0].copy()
                frameR = frame_dict[2].copy()
            else:
                continue  # Skip iteration if frames are not ready

        # Run YOLO detection
        resultsL = model(frameL)
        resultsR = model(frameR)

        ballL, ballR = None, None

        for result in resultsL:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                if class_id == 0:  # If detected object is a ball
                    ballL = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv.rectangle(frameL, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break

        for result in resultsR:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                if class_id == 0:
                    ballR = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv.rectangle(frameR, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break

        # Calculate 3D Coordinates            
        if ballL and ballR:
            disparity = abs(ballL[0] - ballR[0])
            if disparity > 0:
                Z = (FOCAL_LENGTH * BASELINE) / disparity
                X = ((ballL[0] - frameL.shape[1] / 2) * Z) / FOCAL_LENGTH
                Y = ((ballL[1] - frameL.shape[0] / 2) * Z) / FOCAL_LENGTH
                
                with lock:
                    ball_positions.append((X, Y, Z))  # Store detected 3D position
                    processed_frames['ball_3d'] = (X, Y, Z)

        # Update processed frames
        with lock:
            processed_frames[0] = frameL
            processed_frames[2] = frameR

        # FPS Calculation for processing
        curr_time = time.time()
        process_fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

def plot_3d():
    """ Real-time 3D visualization using Matplotlib Animation """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update_plot(frame):
        with lock:
            if ball_positions:
                X, Y, Z = zip(*ball_positions)  # Unpack coordinates
                ax.clear()
                ax.scatter(X, Y, Z, c='red', marker='o')  # Plot points
                ax.plot(X, Y, Z, c='blue')  # Draw trajectory

                ax.set_xlim([-50, 50])
                ax.set_ylim([-50, 50])
                ax.set_zlim([0, 200])  # Adjust based on real-world setup
                ax.set_xlabel('X (cm)')
                ax.set_ylabel('Y (cm)')
                ax.set_zlabel('Z (cm)')
                ax.set_title('3D Ball Position')

    ani = animation.FuncAnimation(fig, update_plot, interval=50)  # Update every 50ms
    plt.show()

def main():
    threadL = threading.Thread(target=test, args=(0,), daemon=True)
    threadR = threading.Thread(target=test, args=(2,), daemon=True)
    threadProcessing = threading.Thread(target=process_frames, daemon=True)
    
    threadL.start()
    threadR.start()
    threadProcessing.start()
    
    # Start 3D plot in the main thread
    plot_3d()

    while True:
        with lock:
            if 0 in frame_dict:
                frameL = frame_dict[0].copy()
                cv.putText(frameL, f"FPS: {fps_dict[0]}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.imshow("Left Camera", frameL)
            
            if 2 in frame_dict:
                frameR = frame_dict[2].copy()
                cv.putText(frameR, f"FPS: {fps_dict[2]}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv.imshow("Right Camera", frameR)
            
        print(f"Processing FPS: {process_fps}")  # Print processing FPS in console
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
