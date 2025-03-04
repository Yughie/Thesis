import cv2
import threading
import os
import time

# Global flags and locks
recording = False
capture_screenshot = {0: False, 2: False}  # Separate flags for each camera
screenshot_count = {0: 1, 2: 1}  # Track screenshot numbering per camera
lock = threading.Lock()
frame_dict = {}  # Store frames for display in the main thread

def capture_frames(camera_id, resolution=(1280, 720), fps=120, save_folder='', screenshot_folder=''):
    global recording, capture_screenshot, frame_dict, screenshot_count
    cap = cv2.VideoCapture(camera_id)

    # Set Camera Properties
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(screenshot_folder, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = None

    frame_count = 0
    last_time = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame from camera {camera_id}")
            break

        frame = cv2.flip(frame, 1)  # Mirror the feed

        # Calculate FPS
        frame_count += 1
        if frame_count >= 10:
            elapsed_time = time.time() - last_time
            fps_display = frame_count / elapsed_time
            frame_count = 0
            last_time = time.time()
            print(f"Camera {camera_id} FPS: {fps_display:.2f}")

        # Overlay FPS
        cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Store frame for main thread display
        with lock:
            frame_dict[camera_id] = frame.copy()

            # Screenshot capture (only once per keypress)
            if capture_screenshot[camera_id]:
                screenshot_filename = os.path.join(screenshot_folder, f"{screenshot_count[camera_id]}.png")

                cv2.imwrite(screenshot_filename, frame)
                print(f"Camera {camera_id}: Screenshot saved: {screenshot_filename}")

                # Increment screenshot count
                screenshot_count[camera_id] += 1

                # Reset flag after saving
                capture_screenshot[camera_id] = False

        # Handle recording
        with lock:
            if recording and video_writer is None:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(save_folder, f"{timestamp}.avi")
                video_writer = cv2.VideoWriter(filename, fourcc, fps, resolution)
                print(f"Camera {camera_id}: Recording started: {filename}")

            if recording and video_writer:
                video_writer.write(frame)

            if not recording and video_writer:
                print(f"Camera {camera_id}: Recording stopped")
                video_writer.release()
                video_writer = None

    cap.release()
    if video_writer:
        video_writer.release()


if __name__ == "__main__":
    camera1_params = {
        'camera_id': 0,
        'resolution': (1280, 720),
        'fps': 120,
        'save_folder': 'left_camera_record',
        'screenshot_folder': 'left_camera_frame'
    }

    camera2_params = {
        'camera_id': 2,
        'resolution': (1280, 720),
        'fps': 120,
        'save_folder': 'right_camera_record',
        'screenshot_folder': 'right_camera_frame'
    }

    thread1 = threading.Thread(target=capture_frames, kwargs=camera1_params, daemon=True)
    thread2 = threading.Thread(target=capture_frames, kwargs=camera2_params, daemon=True)

    thread1.start()
    thread2.start()

    # Main loop for displaying frames and handling key events
    while True:
        with lock:
            if 0 in frame_dict:
                cv2.imshow("Camera 0", frame_dict[0])
            if 2 in frame_dict:
                cv2.imshow("Camera 2", frame_dict[2])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            with lock:
                recording = not recording
                print("Recording toggled:", "ON" if recording else "OFF")
        elif key == ord('c'):
            with lock:
                capture_screenshot[0] = True  # Trigger screenshot for Camera 0
                capture_screenshot[2] = True  # Trigger screenshot for Camera 2
                print("Screenshot capture triggered for both cameras")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
