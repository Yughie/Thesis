import cv2
import os

def capture_calibration_frames_left_camera():
    """
    Captures frames from the left camera at 1280x720 resolution and 120 FPS (if supported)
    and saves them to a folder for calibration purposes.
    """

    # --- Configuration ---
    capture_folder = "./calibration_images/left_cam/" # Modified folder name to reflect settings
    camera_index = 0  # Adjust if your left camera is at a different index
    desired_width = 1280
    desired_height = 720
    desired_fps = 120.0 # Note: FPS is often a float value
    frame_count = 0

    # --- Create capture folder if it doesn't exist ---
    if not os.path.exists(capture_folder):
        os.makedirs(capture_folder)
        print(f"Created folder: {capture_folder}")

    # --- Initialize video capture ---
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}. Please check camera connection.")
        return

    # --- Set desired resolution and FPS ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    # --- Get actual resolution and FPS (to verify if settings were applied) ---
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("--- Left Camera Frame Capture (720p 120fps Attempt) ---")
    print(f"Saving frames to: {capture_folder}")
    print(f"Desired Resolution: {desired_width}x{desired_height}, FPS: {desired_fps}")
    print(f"Actual Resolution:  {int(actual_width)}x{int(actual_height)}, FPS: {actual_fps:.2f} (Check if this matches your desired FPS)") # Display actual FPS with 2 decimal places
    print("Press 'c' or 'C' to capture a frame.")
    print("Press 'q' or 'Q' to quit.")

    cv2.namedWindow("Left Camera Capture")

    while(True):
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame. Exiting.")
            break

        cv2.imshow("Left Camera Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') or key == ord('C'):
            frame_count += 1
            filename = os.path.join(capture_folder, f"left_cam_{frame_count:03d}.jpg") # Modified filename
            cv2.imwrite(filename, frame)
            print(f"Captured frame saved as: {filename}")

        elif key == ord('q') or key == ord('Q'):
            break

    # --- Release resources ---
    cap.release()
    cv2.destroyAllWindows()

    print("Frame capture finished.")

if __name__ == "__main__":
    capture_calibration_frames_left_camera()