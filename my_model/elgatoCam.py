import cv2
import time
import numpy as np

def webcam_fps_test(desired_width=1280, desired_height=720, desired_fps=120):
    """
    Opens the webcam, attempts to set 120 FPS, displays FPS, and records video on 'r' press.
    """
    cap = cv2.VideoCapture(3)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Actual camera FPS (after setting): {actual_fps}")

    if actual_fps != desired_fps:
      print("Warning: Could not set the desired FPS.")

    # Variables for calculating pipeline FPS
    fps_avg_len = 10
    frame_rate_buffer = []
    prev_frame_time = 0

    # Variables for recording
    is_recording = False
    recorder = None
    recording_filename = "output.avi"  # Default filename

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Calculate pipeline FPS
        new_frame_time = time.time()
        instantaneous_fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(instantaneous_fps)
        pipeline_fps = np.mean(frame_rate_buffer)

        # Display text on the frame
        cv2.putText(frame, f"Actual FPS: {actual_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Pipeline FPS: {pipeline_fps:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if is_recording:
            cv2.putText(frame, "RECORDING", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red text
            recorder.write(frame)

        cv2.imshow('Webcam FPS Test', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = not is_recording  # Toggle recording state
            if is_recording:
                print("Recording started...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or 'MJPG', 'DIVX', etc.
                recorder = cv2.VideoWriter(recording_filename, fourcc, actual_fps, (desired_width, desired_height))
            else:
                print("Recording stopped.")
                if recorder:
                  recorder.release()
                  recorder = None


    if recorder:
        recorder.release()  # Release recorder if still active
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_fps_test()