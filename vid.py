from ultralytics import YOLO
import cv2
import os

# Load the YOLOv8x model (ensure GPU is being used if available)
model = YOLO('C:\\Users\\hp\\Downloads\\football_analysis-main\\besttt.pt')

# Path to the input video
input_video_path = 'mo.mp4'  # Replace with your actual video path
output_video_path = 'C:\\Users\\hp\\Downloads\\footballanalysis-main\\output_video.mp4'

# Create a VideoCapture object to read the video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Frame skipping factor (e.g., process every 2nd frame)
frame_skip = 10  # Change this value to control how many frames to skip

# Loop through the video frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop when the video ends

    # Skip frames to speed up processing
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Optional: Resize frame to speed up processing (e.g., to 50% size)
    resized_frame = cv2.resize(frame, (width // 2, height // 2))

    # Perform detection on the resized frame
    results = model(resized_frame)
    
    # Extract the detected boxes, classes, and confidence scores
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Box coordinates in [x1, y1, x2, y2] format
        classes = result.boxes.cls.cpu().numpy()  # Class labels
        
        # Process detected boxes and classes
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            class_name = result.names[int(cls)]  # Get the actual class name
            
            # Adjust coordinates back to the original size if resized
            x1, y1, x2, y2 = x1 * 2, y1 * 2, x2 * 2, y2 * 2
            
            # Filter to process all relevant classes: ball, player, referee, goalkeeper
            if class_name in ['ball', 'player', 'referee', 'goalkeeper']:
                # Draw the bounding box on the original frame
                color = (0, 255, 0) if class_name == 'player' else (0, 0, 255) if class_name == 'ball' else (255, 0, 0) if class_name == 'referee' else (255, 255, 0)  # Colors for each class
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the frame in a window
    cv2.imshow('Video Processing', frame)

    # Write the frame with detections to the output video
    out.write(frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release video objects and close windows
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_video_path}")
