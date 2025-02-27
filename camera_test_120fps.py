import cv2
import time
import numpy as np
from ultralytics import YOLO

usb_idx = 0  # Camera index
resW, resH = 640, 480  # Resolution
model_path = 'my_model.pt'  # Path to your YOLO model (MAKE SURE THIS IS CORRECT)

# Load YOLO model
try:
    model = YOLO(model_path, task='detect').to('cuda') # Or .to('cpu') if no GPU
    print(f"YOLO model loaded from: {model_path}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

cap = cv2.VideoCapture(usb_idx)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# --- Force MJPG and Attempt 120 FPS ---
desired_fps = 120
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, desired_fps)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Actual Camera FPS from cap.get(cv2.CAP_PROP_FPS): {actual_fps}")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

frame_rate_buffer = []
fps_avg_len = 200
avg_frame_rate = 0

print("Starting capture and YOLO inference FPS measurement...")

is_recording = False
video_writer = None
output_filename = "output_detection.avi" # Output filename for recording

while True:
    t_start_frame = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # --- YOLO INFERENCE ---
    t_start_inference = time.perf_counter()
    results = model(frame, verbose=False) # Run inference (on original 640x480 frame for now)
    t_stop_inference = time.perf_counter()
    inference_time = t_stop_inference - t_start_inference

    # --- DRAWING ANNOTATIONS ---
    annotated_frame = frame.copy() # Create a copy to draw on, keep original frame for FPS

    detections = results[0].boxes
    if detections: # Check if any detections were made
        for detection in detections:
            xyxy = detection.xyxy[0].cpu().numpy().astype(int) # Get bounding box coordinates (xyxy format)
            conf = detection.conf[0].cpu().numpy() # Get confidence score
            class_id = int(detection.cls[0].cpu().numpy()) # Get class ID
            class_name = model.names[class_id] # Get class name from model's names

            # Draw bounding box
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2) # Green box

            # Create label string (e.g., "person 0.95")
            label = f"{class_name} {conf:.2f}"

            # Calculate text position (slightly above the top-left corner of the box)
            text_x = xyxy[0]
            text_y = xyxy[1] - 10 if xyxy[1] - 10 > 10 else xyxy[1] + 10 # Adjust y position if too close to top

            # Draw class label
            cv2.putText(annotated_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Green text


    # --- Recording logic ---
    key = cv2.waitKey(1)
    if key == ord('r'):
        if not is_recording:
            print("Starting recording...")
            is_recording = True
            fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Use MJPG for recording, same as capture - TRY THIS FIRST
            # fourcc = cv2.VideoWriter_fourcc(*'XVID') # You can try XVID or other codecs if MJPG fails
            record_fps = 30.0 # Default FPS for recording if actual_fps is not reliable

            if actual_fps > 0 and actual_fps < 1000: # Sanity check for actual_fps
                record_fps = actual_fps
            else:
                print(f"Warning: actual_fps ({actual_fps}) is invalid, defaulting to 30 FPS for recording.")

            video_writer = cv2.VideoWriter(output_filename, fourcc, record_fps, (resW, resH)) # Use annotated_frame resolution
            if not video_writer.isOpened():
                print("Error: Could not open video writer")
                is_recording = False # Stop trying to record
            else:
                print(f"Video writer opened with FPS: {record_fps}, fourcc: MJPG") # Debug message
        else:
            print("Stopping recording...")
            is_recording = False
            if video_writer:
                video_writer.release()
                video_writer = None # Reset video_writer

    if is_recording:
        if video_writer:
            video_writer.write(annotated_frame) # Write the annotated frame

    # --- DISPLAY ANNOTATED FRAME ---
    cv2.imshow("Camera Feed", annotated_frame) # Display the annotated frame


    # FPS calculation
    t_stop_frame = time.perf_counter()
    frame_time = t_stop_frame - t_start_frame  # Time for the entire loop
    frame_rate_calc = 1.0 / frame_time if frame_time > 0 else 0  # Avoid division by zero
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

    print(f'Inference Time: {inference_time:.4f}s, Frame Time: {frame_time:.4f}s, Pipeline FPS (Capture + Inference): {avg_frame_rate:.2f} {"(REC)" if is_recording else ""}', end='\r')


    if key == ord('q'):
        break

# Release resources
cap.release()
if video_writer: # Release video writer if it was initialized and recording was stopped by 'q'
    video_writer.release()
cv2.destroyAllWindows()
print(f"\nAverage Pipeline FPS (Capture + Inference): {avg_frame_rate:.2f}")
print("Camera and YOLO resources released.")