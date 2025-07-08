import numpy as np
import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames 


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def draw_bezier_pass(frame, start, end, color, thickness=2, curve_height=60):
    # Calculate control point for the curve
    start = np.array(start)
    end = np.array(end)
    mid = (start + end) / 2
    # Perpendicular direction for the curve
    direction = end - start
    perp = np.array([-direction[1], direction[0]])
    norm = np.linalg.norm(perp)
    perp = perp / norm if norm != 0 else perp
    control = mid + perp * curve_height

    # Generate points along the Bézier curve
    points = []
    for t in np.linspace(0, 1, 30):
        point = (1-t)**2 * start + 2*(1-t)*t * control + t**2 * end
        points.append(point.astype(int))
    points = np.array(points).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], False, color, thickness)
    # Draw an arrowhead at the end
    if len(points) > 1:
        cv2.arrowedLine(frame, tuple(points[-2][0]), tuple(points[-1][0]), color, thickness, tipLength=0.3)