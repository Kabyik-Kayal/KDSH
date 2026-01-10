"""
Training module for TextPath consistency classification.
Provides Trainer class and training utilities.
"""

from .trainer import TrainingConfig, Trainer, run_training
from .pretraining import run_pretraining, run_pretraining_from_config

__all__ = [
    'TrainingConfig', 
    'Trainer', 
    'run_training',
    'run_pretraining',
    'run_pretraining_from_config'
]
