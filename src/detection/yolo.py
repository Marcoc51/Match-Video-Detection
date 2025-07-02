from ultralytics import YOLO

from pathlib import Path

# Go up from yolo.py to project root
project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "models" / "yolo" / "best.pt"

model = YOLO(str(model_path))

def detect_objects_in_frame(frame):
    results = model.predict(frame, save=False)
    return results[0]  # Return the first result (for the frame)

# results = model.predict("input_videos/08fd33_4.mp4", save=True)

# print(results[0])
# print("=============================")
# for box in results[0].boxes:
#     print(box)