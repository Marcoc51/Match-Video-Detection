"""
Training module for cross detection model fine-tuning.
"""

from .data_preparation import CrossDataPreparation
from .model_deployment import CrossDetectionModelDeployment

__all__ = [
    'CrossDataPreparation',
    'CrossDetectionModelDeployment'
] 