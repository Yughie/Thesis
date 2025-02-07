import cv2
import torch
import argparse # For resolution
import time

from ultralytics import YOLO
import supervision as sv

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration of resolution 
def parse_arguments() -> argparse.Namespace: 
    parser = argparse.ArgumentParser(description="YOLOv8 Live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1024,768],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(1) # Get the feed of camera 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Retrieve actual FPS
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Actual FPS: {actual_fps}")

    model = YOLO("yolo11n.pt")
    model.to(device)

    class_list = model.names
    print(class_list)
          
    circle_annotator = sv.CircleAnnotator(
        thickness=2
    )

    fps = 0
    frame_count = 0
    prev_time = time.time()


    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        frame = circle_annotator.annotate(scene=frame, detections=detections)

        # Iterate through detections to add labels
        for detection in detections:
            # Assuming detection is a tuple (bbox, class_id, confidence)
            print(f"Detection: {detection}")
            bbox, class_idx, confidence, *rest = detection  # Unpack the tuple
            
            print(f"bbox {bbox}")
            print(f"id {class_idx}")
            print(f"confidence {confidence}")


            if class_idx is not None:
                # Get the class label from the model's class names
                class_label = model.names.get(class_idx, "Unknown")
            
                # Define the position for the label (top-left corner of the bounding box)
                label_position = (int(bbox[0]), int(bbox[1]) - 10)

                # Create the label text
                label_text = f"{class_label} ({confidence:.2f})"

                # Add the label text to the frame
                cv2.putText(frame, label_text, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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