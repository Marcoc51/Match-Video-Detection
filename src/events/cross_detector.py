import cv2
import numpy as np
import os
import shutil
from sklearn.cluster import KMeans
from collections import defaultdict
from pathlib import Path

# Base directory where images are located
project_root = Path(__file__).resolve().parents[2]
base_dir = project_root / "Match-Video-Detection" / "outputs"

# Output directories for categorized images
team1_dir = os.path.join(base_dir, 'Team1')
team2_dir = os.path.join(base_dir, 'Team2')
ball_dir = os.path.join(base_dir, 'Ball')
others_dir = os.path.join(base_dir, 'Others')

# Create directories if they don't exist
os.makedirs(team1_dir, exist_ok=True)
os.makedirs(team2_dir, exist_ok=True)
os.makedirs(ball_dir, exist_ok=True)
os.makedirs(others_dir, exist_ok=True)

# Supported image formats
supported_formats = ['.jpg', '.jpeg', '.png']

# Function to detect the ball (white and circular object)
def detect_ball(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Apply HoughCircles to detect circles (the ball)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=5, maxRadius=50)
    
    # Check if any circles were detected
    if circles is not None:
        return True  # Ball detected
    return False

# Function to extract the dominant color from an image
def get_dominant_color(image, k=1):
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply KMeans to find the most dominant color
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Return the dominant color
    return kmeans.cluster_centers_[0]

# Initialize list to hold features (dominant colors) and file paths
features = []
image_paths = []
ball_images = []

# Iterate through all files in the directory
for filename in os.listdir(base_dir):
    # Check if the file is an image
    if any(filename.lower().endswith(ext) for ext in supported_formats):
        img_path = os.path.join(base_dir, filename)
        
        # Ensure the file exists before processing
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            if image is not None:
                # First, check if it's the ball
                if detect_ball(image):
                    ball_images.append(img_path)
                else:
                    # Resize image to speed up processing
                    resized_image = cv2.resize(image, (100, 100))
                    
                    # Get dominant color
                    dominant_color = get_dominant_color(resized_image)
                    features.append(dominant_color)
                    image_paths.append(img_path)
        else:
            print(f"Warning: {img_path} does not exist. Skipping.")

# Convert to numpy array for KMeans
if features:
    features = np.array(features)

    # Use KMeans to cluster images into 2 groups (Team1, Team2)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features)

    # Dictionary to store clusters
    cluster_dict = defaultdict(list)

    # Assign images to clusters
    for label, img_path in zip(labels, image_paths):
        cluster_dict[label].append(img_path)

    # Move images based on clusters
    for cluster, img_paths in cluster_dict.items():
        if cluster == 0:
            dest_dir = team1_dir
        elif cluster == 1:
            dest_dir = team2_dir
        
        for img_path in img_paths:
            try:
                shutil.move(img_path, os.path.join(dest_dir, os.path.basename(img_path)))
            except FileNotFoundError as e:
                print(f"Error: {e}. Skipping the file.")

# Move ball images to the ball directory
for img_path in ball_images:
    try:
        shutil.move(img_path, os.path.join(ball_dir, os.path.basename(img_path)))
    except FileNotFoundError as e:
        print(f"Error: {e}. Skipping the file.")

print("Images have been automatically categorized and moved to their respective folders.")

class CrossDetector:
    def __init__(self, field_width, field_height, penalty_area_coords):
        """
        field_width, field_height: dimensions of the field in pixels or meters
        penalty_area_coords: ((x1, y1), (x2, y2)) top-left and bottom-right of penalty area
        """
        self.field_width = field_width
        self.field_height = field_height
        self.penalty_area_coords = penalty_area_coords

    def is_in_wing(self, ball_pos):
        # Define wing as left/right 20% of the field width
        x, y = ball_pos
        return x < 0.2 * self.field_width or x > 0.8 * self.field_width

    def is_in_penalty_area(self, ball_pos):
        (x1, y1), (x2, y2) = self.penalty_area_coords
        x, y = ball_pos
        return x1 <= x <= x2 and y1 <= y <= y2

    def detect_crosses(self, ball_trajectory):
        """
        ball_trajectory: list of (frame_idx, (x, y)) tuples
        Returns: list of (start_frame, end_frame) for detected crosses
        """
        crosses = []
        in_wing = False
        cross_start = None

        for idx, (frame, pos) in enumerate(ball_trajectory):
            if self.is_in_wing(pos):
                if not in_wing:
                    in_wing = True
                    cross_start = frame
            elif in_wing and self.is_in_penalty_area(pos):
                crosses.append((cross_start, frame))
                in_wing = False
                cross_start = None
            else:
                in_wing = False
                cross_start = None
        return crosses
