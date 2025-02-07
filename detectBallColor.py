import cv2
import numpy as np

# Callback function for trackbar (does nothing but required)
def nothing(x):
    pass

# Open video capture (or replace with image)
cap = cv2.VideoCapture(0)  

# Create a window
cv2.namedWindow("Trackbars")

# Create trackbars for Hue, Saturation, and Value
cv2.createTrackbar("Lower Hue", "Trackbars", 5, 180, nothing)
cv2.createTrackbar("Upper Hue", "Trackbars", 25, 180, nothing)
cv2.createTrackbar("Lower Sat", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("Upper Sat", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Lower Val", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("Upper Val", "Trackbars", 255, 255, nothing)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current trackbar positions
    l_h = cv2.getTrackbarPos("Lower Hue", "Trackbars")
    u_h = cv2.getTrackbarPos("Upper Hue", "Trackbars")
    l_s = cv2.getTrackbarPos("Lower Sat", "Trackbars")
    u_s = cv2.getTrackbarPos("Upper Sat", "Trackbars")
    l_v = cv2.getTrackbarPos("Lower Val", "Trackbars")
    u_v = cv2.getTrackbarPos("Upper Val", "Trackbars")

    # Define the range
    lower_orange = np.array([l_h, l_s, l_v])
    upper_orange = np.array([u_h, u_s, u_v])

    # Threshold the image
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Bitwise-AND to extract the orange object
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show results
    cv2.imshow("Mask", mask)
    cv2.imshow("Detected Object", result)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
