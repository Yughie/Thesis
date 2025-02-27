import cv2
import time
import numpy as np
from ultralytics import YOLO
import threading
import queue

usb_idx = 0
resW, resH = 1280, 720
model_path = 'my_model.pt'

# --- Frame Buffer Queue ---
frame_queue = queue.Queue(maxsize=5)  # Bounded queue to prevent excessive memory usage

# --- Load YOLO model (load once outside threads) ---
try:
    model = YOLO(model_path, task='detect').to('cuda')
    print(f"YOLO model loaded from: {model_path}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

cap = cv2.VideoCapture(usb_idx)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# --- Set Camera Properties ---
desired_fps = 30
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, desired_fps)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Actual Camera FPS from cap.get(cv2.CAP_PROP_FPS): {actual_fps}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# --- Variables for Recording ---
recording = False
video_writer = None
output_filename = "output_threaded.avi" # Different output filename for threaded version

frame_rate_buffer = []
fps_avg_len = 200
avg_frame_rate = 0
stop_event = threading.Event() # Event to signal threads to stop

# --- Function for Capture Thread (Producer) ---
def capture_frames(capture, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = capture.read()
        if not ret:
            print("Error: Capture thread - Can't receive frame (stream end?).")
            stop_event.set() # Signal other threads to stop
            break
        try:
            frame_queue.put(frame, timeout=0.01) # Non-blocking put with timeout
        except queue.Full:
            print("Warning: Frame queue is full, dropping frame from capture.") # Queue is full, consumer is too slow

    capture.release() # Release camera in capture thread when stopped
    print("Capture thread stopped.")


# --- Start Capture Thread ---
capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue, stop_event))
capture_thread.daemon = True # Allow main thread to exit even if capture thread is running
capture_thread.start()

print("Press 'R' to start/stop recording. Press 'Q' to quit.")
time.sleep(2) # Give capture thread time to fill queue

# --- Main Processing Loop (Consumer - Main Thread) ---
while True:
    t_start_frame = time.perf_counter()

    try:
        frame = frame_queue.get(timeout=0.01) # Non-blocking get with timeout
    except queue.Empty:
        if stop_event.is_set(): # Check if capture thread stopped due to error
            print("Exiting processing loop due to capture thread error or stop signal.")
            break
        print("Warning: Frame queue is empty, waiting for frame...") # Consumer is faster than producer temporarily
        continue # Wait for frame from queue

    # --- YOLO INFERENCE ---
    t_start_inference = time.perf_counter()
    results = model(frame, verbose=False)
    t_stop_inference = time.perf_counter()
    inference_time = t_stop_inference - t_start_inference

    # --- DRAWING DETECTIONS ---
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # --- RECORDING LOGIC ---
    if recording:
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_writer = cv2.VideoWriter(output_filename, fourcc, actual_fps, (resW, resH))
            print("Recording started (threaded)...")
        video_writer.write(frame)

    # --- DISPLAY FRAME ---
    cv2.imshow("Camera Feed (Threaded)", frame) # Different window title for threaded version

    # FPS calculation
    t_stop_frame = time.perf_counter()
    frame_time = t_stop_frame - t_start_frame
    frame_rate_calc = 1.0 / frame_time if frame_time > 0 else 0
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

    print(f'Inference Time: {inference_time:.4f}s, Frame Time: {frame_time:.4f}s, Pipeline FPS (Threaded): {avg_frame_rate:.2f}', end='\r')

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        recording = not recording
        if not recording and video_writer is not None:
            video_writer.release()
            video_writer = None
            print("\nRecording stopped (threaded). Video saved.")
    elif key == ord('q'):
        stop_event.set() # Signal capture thread to stop
        break

# --- Cleanup ---
stop_event.set() # Ensure capture thread stops if main loop quits
capture_thread.join(timeout=5) # Wait for capture thread to finish (with timeout)
if capture_thread.is_alive():
    print("Warning: Capture thread did not terminate gracefully.")

if video_writer is not None:
    video_writer.release()
cap.release() # Release camera if capture thread didn't already (redundant but safe)
cv2.destroyAllWindows()

print(f"\nAverage Pipeline FPS (Threaded): {avg_frame_rate:.2f}")
print(f"Final recording saved as {output_filename}" if recording else "No recording saved.")
print("Camera and YOLO resources released (threaded version).")