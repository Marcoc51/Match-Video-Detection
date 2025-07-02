from ultralytics import YOLO
import cv2
import os

# Load the YOLOv8x model
model = YOLO('C:\\Users\\hp\\Downloads\\football_analysis-main\\besttt.pt')

# Create a directory to save the images with detections
output_dir = 'C:\\Users\\hp\\Downloads\\footballanalysis-main\\out'
os.makedirs(output_dir, exist_ok=True)

# List of images to process
image_paths = ['t.jpg']  # Replace with your actual image paths

# Loop through each image
for i, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    
    # Check if the image is loaded correctly
    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    # Perform detection
    results = model(image)
    
    # Extract the detected boxes, classes, and confidence scores
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Box coordinates in [x1, y1, x2, y2] format
        classes = result.boxes.cls.cpu().numpy()  # Class labels
        
        # Process detected boxes and classes
        for j, (box, cls) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            class_name = result.names[int(cls)]  # Get the actual class name
            
            # Filter to process all relevant classes: ball, players, referees
            if class_name in ['ball', 'player', 'referee']:
                print(f"Processing and saving {class_name}")

                # Draw the bounding box on the original image
                color = (0, 255, 0) if class_name == 'players' else (0, 0, 255) if class_name == 'ball' else (255, 0, 0)  # Green for players, Red for ball, Blue for referees
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Crop the detected region from the image
                cropped_img = image[y1:y2, x1:x2]

                # Check if the crop is valid
                if cropped_img.size == 0:
                    print(f"Invalid crop for {class_name} at {x1}, {y1}, {x2}, {y2}")
                    continue

                # Save the cropped image
                cropped_img_name = f"{class_name}_{i+1}_{j+1}.jpg"
                output_path = os.path.join(output_dir, cropped_img_name)
                
                # Save and check if the write was successful
                success = cv2.imwrite(output_path, cropped_img)
                if success:
                    print(f"Saved {cropped_img_name} to {output_path}")
                else:
                    print(f"Failed to save {cropped_img_name} to {output_path}")

    # Save the image with bounding boxes and references
    referenced_image_path = os.path.join(output_dir, f"referenced_{i+1}.jpg")
    cv2.imwrite(referenced_image_path, image)
    print(f"Saved referenced image with bounding boxes to {referenced_image_path}")

print(f"Cropped images and images with references are saved in '{output_dir}' directory.")
