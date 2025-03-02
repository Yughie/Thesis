import cv2

def test_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Camera at index {camera_index} is available.")
        cap.release()
        return True
    else:
        print(f"Camera at index {camera_index} is NOT available.")
        return False

if __name__ == "__main__":
    print("Probing camera indices...")
    for i in range(10): # Test indices 0 to 9 (adjust range if you suspect more)
        test_camera(i)
    print("Camera probing finished.")