import cv2
from ultralytics import YOLO

# Load the YOLOv5 model from a local file and specify the class to detect (sports ball, class index 32)
model = YOLO("yolo11n.pt")
model.classes = [32]

# Initialize webcam capture
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 640x480 for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Perform object detection using YOLOv5
    results = model(frame)

    # Ensure results are not empty
    if results[0].boxes is not None:
        # Get the detected boxes
        boxes = results[0].boxes.cpu().numpy()

        # Loop through each detected object
        for box in boxes:
            if box[4] == 32:  # Class index for sports ball
                x1, y1, x2, y2, conf, class_idx = box[:6]  # Unpack the first 6 values
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_name = class_list[int(class_idx)]
                confidence = conf

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put label above the bounding box
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow("YOLOv5 Object Detection", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
