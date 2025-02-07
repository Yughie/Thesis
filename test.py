import cv2
import numpy as np

cap = cv2.VideoCapture(2)  # Change index if needed

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Attempt to set a high FPS
cap.set(cv2.CAP_PROP_FPS, 120)  

# Set resolution
width, height = 1080, 640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Retrieve actual FPS
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Actual FPS: {actual_fps}")
print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# Define HSV color range for orange (tweak if needed)
lower_orange = np.array([5, 47, 108])  # Adjust hue, saturation, value
upper_orange = np.array([255, 141, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Find contours of the detected ball
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small noise
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            if radius > 10:  # Minimum size to be considered
                cv2.circle(frame, center, radius, (0, 255, 0), 2)  # Draw circle
                cv2.putText(frame, "Ball Detected", (center[0] - 50, center[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display FPS on frame
    cv2.putText(frame, f"FPS: {actual_fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show results
    cv2.imshow("Table Tennis Ball Detection", frame)
    cv2.imshow("Mask", mask)  # Show mask for debugging

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
