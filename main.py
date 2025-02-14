import cv2
import torch
import argparse # For resolution
import time

from ultralytics import YOLO

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = YOLO("yolo11n.pt")
model.classes = [32]
model.to(device)

# Configuration of resolution 
def parse_arguments() -> argparse.Namespace: 
    parser = argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument(
        "--webcam-resolution",
        default=[640, 480],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0) # Get the feed of camera 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Retrieve actual FPS
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Actual FPS: {actual_fps}")

    class_list = model.names
    print(class_list)

    fps = 0
    frame_count = 0
    prev_time = time.time()


    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.track(frame, persist=True)
        #print(results)

        #Ensure results are not empty
        if results[0].boxes.data is not None:
            # Get the detected boxes, their class indices, and confidences
            boxes = results[0].boxes.data.numpy()
            class_indices = results[0].boxes.cls.numpy().astype(int)
            confidences = results[0].boxes.conf.numpy()
            
            # Loop through each detected object
            
            for box, class_idx, conf in zip(boxes, class_indices, confidences):

                x1, y1, x2, y2 = map(int, box[:4])  # Unpack the first 4 values as integers

                # Calculate the Center of the Box
                cx = (x1 + x2) // 2 
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                class_name = class_list[class_idx]
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Put label above the bounding box
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    

        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:  # Update FPS every 10 frames
            curr_time = time.time()
            fps = frame_count / (curr_time - prev_time)
            prev_time = curr_time
            frame_count = 0

        # Display FPS on the frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        cv2.imshow("yolov11", frame)

       #print(frame.shape) # Show resolution
        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()