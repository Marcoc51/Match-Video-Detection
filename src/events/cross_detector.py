"""
Cross detection logic for football event analytics.
Detects crosses as passes that start in wide areas and end in the penalty area.
"""
import numpy as np

class CrossDetector:
    @staticmethod
    def detect_crosses(passes, frame_shape, wide_ratio=0.2, penalty_y_ratio=0.8, min_length=50):
        """
        Detect crosses from a list of passes.
        Args:
            passes: list of pass objects with start_ball_bbox and end_ball_bbox
            frame_shape: (height, width) of the video frame
            wide_ratio: fraction of width considered as wide area (default 0.2)
            penalty_y_ratio: fraction of height considered as penalty area (default 0.8)
            min_length: minimum pass length to be considered a cross
        Returns:
            List of passes that are crosses
        """
        h, w = frame_shape[:2]
        wide_margin = int(wide_ratio * w)
        penalty_y_min = int(penalty_y_ratio * h)
        crosses = []
        for p in passes:
            # Get start and end positions
            start = np.array([
                (p.start_ball_bbox[0] + p.start_ball_bbox[2]) / 2,
                (p.start_ball_bbox[1] + p.start_ball_bbox[3]) / 2
            ])
            end = np.array([
                (p.end_ball_bbox[0] + p.end_ball_bbox[2]) / 2,
                (p.end_ball_bbox[1] + p.end_ball_bbox[3]) / 2
            ])
            # Start in wide area
            if not (start[0] < wide_margin or start[0] > w - wide_margin):
                continue
            # End in penalty area (vertical only, can be refined)
            if not (end[1] > penalty_y_min):
                continue
            # Minimum length
            if np.linalg.norm(end - start) < min_length:
                continue
            crosses.append(p)
        return crosses
