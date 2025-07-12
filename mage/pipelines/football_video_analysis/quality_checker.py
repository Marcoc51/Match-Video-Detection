"""
Quality Checker Block for Football Video Analysis Pipeline
Handles quality validation and assessment of detection results
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager

logger = logging.getLogger(__name__)


@transformer
def check_quality(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    Check quality of detection results.
    
    This block handles:
    - Detection quality assessment
    - Confidence validation
    - Coverage analysis
    - Quality scoring and recommendations
    
    Args:
        data: Input data containing detection results
        
    Returns:
        Dict containing quality assessment results
    """
    try:
        detections = data['detections']
        job_id = data.get('job_id')
        
        # Update job progress if available
        if job_id:
            job_manager = get_job_manager()
            job_manager.update_job(job_id, progress=40, message="Checking quality...")
        
        logger.info("Starting quality assessment")
        
        # Perform quality checks
        quality_results = {
            'detection_quality': assess_detection_quality(detections),
            'confidence_analysis': analyze_confidence(detections),
            'coverage_analysis': analyze_coverage(detections),
            'quality_recommendations': generate_quality_recommendations(detections),
            'overall_quality_score': 0.0
        }
        
        # Calculate overall quality score
        quality_results['overall_quality_score'] = calculate_overall_quality(quality_results)
        
        # Combine results
        result = {
            **data,
            'quality_assessment': quality_results
        }
        
        logger.info(f"Quality assessment completed. Overall score: {quality_results['overall_quality_score']:.2f}")
        return result
        
    except Exception as e:
        logger.error(f"Error in quality checker: {e}")
        raise


def assess_detection_quality(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assess the quality of detections."""
    if not detections:
        return {
            'score': 0.0,
            'level': 'poor',
            'issues': ['no_detections'],
            'total_frames': 0,
            'frames_with_detections': 0
        }
    
    total_frames = len(detections)
    frames_with_detections = sum(1 for d in detections if d.get('detection_count', 0) > 0)
    
    # Calculate coverage ratio
    coverage_ratio = frames_with_detections / total_frames if total_frames > 0 else 0
    
    # Calculate average detections per frame
    total_detections = sum(d.get('detection_count', 0) for d in detections)
    avg_detections = total_detections / total_frames if total_frames > 0 else 0
    
    # Calculate detection consistency
    detection_counts = [d.get('detection_count', 0) for d in detections]
    consistency = 1 - (np.std(detection_counts) / (np.mean(detection_counts) + 1e-6))
    consistency = max(0, min(1, consistency))  # Clamp between 0 and 1
    
    # Calculate quality score
    quality_score = (coverage_ratio * 0.4 + consistency * 0.3 + min(avg_detections / 10, 1) * 0.3)
    
    # Determine quality level
    if quality_score >= 0.8:
        level = 'excellent'
    elif quality_score >= 0.6:
        level = 'good'
    elif quality_score >= 0.4:
        level = 'fair'
    else:
        level = 'poor'
    
    # Identify issues
    issues = []
    if coverage_ratio < 0.5:
        issues.append('low_coverage')
    if consistency < 0.5:
        issues.append('inconsistent_detections')
    if avg_detections < 2:
        issues.append('insufficient_detections')
    
    return {
        'score': quality_score,
        'level': level,
        'issues': issues,
        'total_frames': total_frames,
        'frames_with_detections': frames_with_detections,
        'coverage_ratio': coverage_ratio,
        'average_detections_per_frame': avg_detections,
        'detection_consistency': consistency
    }


def analyze_confidence(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze confidence scores of detections."""
    all_confidences = []
    
    for frame_detection in detections:
        for detection in frame_detection.get('detections', []):
            confidence = detection.get('confidence', 0)
            all_confidences.append(confidence)
    
    if not all_confidences:
        return {
            'average_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0,
            'confidence_distribution': {},
            'low_confidence_count': 0
        }
    
    confidences = np.array(all_confidences)
    
    # Calculate confidence statistics
    avg_confidence = np.mean(confidences)
    min_confidence = np.min(confidences)
    max_confidence = np.max(confidences)
    
    # Analyze confidence distribution
    confidence_bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    confidence_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
    distribution = {}
    for i in range(len(confidence_bins) - 1):
        count = np.sum((confidences >= confidence_bins[i]) & (confidences < confidence_bins[i + 1]))
        distribution[confidence_labels[i]] = int(count)
    
    # Count low confidence detections
    low_confidence_count = np.sum(confidences < 0.5)
    
    return {
        'average_confidence': float(avg_confidence),
        'min_confidence': float(min_confidence),
        'max_confidence': float(max_confidence),
        'confidence_distribution': distribution,
        'low_confidence_count': int(low_confidence_count),
        'low_confidence_ratio': float(low_confidence_count / len(confidences))
    }


def analyze_coverage(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze detection coverage across the video."""
    if not detections:
        return {
            'temporal_coverage': 0.0,
            'spatial_coverage': {},
            'coverage_gaps': []
        }
    
    # Analyze temporal coverage
    total_frames = len(detections)
    frames_with_detections = sum(1 for d in detections if d.get('detection_count', 0) > 0)
    temporal_coverage = frames_with_detections / total_frames
    
    # Analyze spatial coverage (simplified)
    all_bboxes = []
    for frame_detection in detections:
        for detection in frame_detection.get('detections', []):
            bbox = detection.get('bbox', [0, 0, 0, 0])
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            all_bboxes.append([center_x, center_y])
    
    spatial_coverage = {}
    if all_bboxes:
        bboxes = np.array(all_bboxes)
        
        # Calculate coverage by quadrants
        width, height = 1920, 1080  # Assuming standard resolution
        quadrants = {
            'top_left': np.sum((bboxes[:, 0] < width/2) & (bboxes[:, 1] < height/2)),
            'top_right': np.sum((bboxes[:, 0] >= width/2) & (bboxes[:, 1] < height/2)),
            'bottom_left': np.sum((bboxes[:, 0] < width/2) & (bboxes[:, 1] >= height/2)),
            'bottom_right': np.sum((bboxes[:, 0] >= width/2) & (bboxes[:, 1] >= height/2))
        }
        
        spatial_coverage = {k: int(v) for k, v in quadrants.items()}
    
    # Find coverage gaps
    coverage_gaps = find_coverage_gaps(detections)
    
    return {
        'temporal_coverage': temporal_coverage,
        'spatial_coverage': spatial_coverage,
        'coverage_gaps': coverage_gaps
    }


def find_coverage_gaps(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find gaps in detection coverage."""
    gaps = []
    
    for i in range(1, len(detections)):
        prev_frame = detections[i-1]
        curr_frame = detections[i]
        
        prev_count = prev_frame.get('detection_count', 0)
        curr_count = curr_frame.get('detection_count', 0)
        
        # If there's a significant drop in detections
        if prev_count > 0 and curr_count == 0:
            gaps.append({
                'start_frame': i-1,
                'end_frame': i,
                'gap_size': 1,
                'severity': 'minor' if i < len(detections) - 1 and detections[i+1].get('detection_count', 0) > 0 else 'major'
            })
    
    return gaps


def generate_quality_recommendations(detections: List[Dict[str, Any]]) -> List[str]:
    """Generate quality improvement recommendations."""
    recommendations = []
    
    if not detections:
        recommendations.append("No detections found. Check video quality and model performance.")
        return recommendations
    
    # Analyze detection patterns
    total_frames = len(detections)
    frames_with_detections = sum(1 for d in detections if d.get('detection_count', 0) > 0)
    coverage_ratio = frames_with_detections / total_frames
    
    # Coverage recommendations
    if coverage_ratio < 0.5:
        recommendations.append("Low detection coverage detected. Consider adjusting confidence thresholds or improving video quality.")
    
    if coverage_ratio < 0.3:
        recommendations.append("Very low coverage. Model may need retraining or video preprocessing.")
    
    # Confidence recommendations
    all_confidences = []
    for frame_detection in detections:
        for detection in frame_detection.get('detections', []):
            all_confidences.append(detection.get('confidence', 0))
    
    if all_confidences:
        avg_confidence = np.mean(all_confidences)
        low_confidence_ratio = np.sum(np.array(all_confidences) < 0.5) / len(all_confidences)
        
        if avg_confidence < 0.6:
            recommendations.append("Low average confidence. Consider model retraining or threshold adjustment.")
        
        if low_confidence_ratio > 0.3:
            recommendations.append("High proportion of low-confidence detections. Review model performance.")
    
    # Consistency recommendations
    detection_counts = [d.get('detection_count', 0) for d in detections]
    consistency = 1 - (np.std(detection_counts) / (np.mean(detection_counts) + 1e-6))
    
    if consistency < 0.5:
        recommendations.append("Inconsistent detection patterns. Check for video quality issues or camera movement.")
    
    # General recommendations
    if len(recommendations) == 0:
        recommendations.append("Detection quality is good. No immediate improvements needed.")
    
    return recommendations


def calculate_overall_quality(quality_results: Dict[str, Any]) -> float:
    """Calculate overall quality score."""
    detection_quality = quality_results['detection_quality']
    confidence_analysis = quality_results['confidence_analysis']
    
    # Weighted combination of different quality metrics
    detection_score = detection_quality.get('score', 0)
    confidence_score = confidence_analysis.get('average_confidence', 0)
    coverage_score = detection_quality.get('coverage_ratio', 0)
    
    # Calculate overall score
    overall_score = (
        detection_score * 0.4 +
        confidence_score * 0.3 +
        coverage_score * 0.3
    )
    
    return float(overall_score)


@test
def test_quality_checker():
    """Test the quality checker with sample data."""
    # Create sample detection data
    sample_detections = [
        {
            'frame_index': 0,
            'detection_count': 2,
            'detections': [
                {'confidence': 0.9, 'bbox': [100, 100, 120, 140]},
                {'confidence': 0.8, 'bbox': [150, 150, 160, 160]}
            ]
        },
        {
            'frame_index': 1,
            'detection_count': 1,
            'detections': [
                {'confidence': 0.85, 'bbox': [110, 110, 130, 150]}
            ]
        }
    ]
    
    # Test quality assessment
    quality = assess_detection_quality(sample_detections)
    
    assert 'score' in quality
    assert 'level' in quality
    assert 'issues' in quality
    assert quality['total_frames'] == 2
    
    print("✅ Quality checker test passed") 