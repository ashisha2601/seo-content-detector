"""
Utility modules for SEO Content Quality & Duplicate Detector
"""

from .parser import parse_dataset, extract_text_from_html
from .features import FeatureExtractor
from .scorer import (
    detect_duplicates,
    detect_thin_content,
    create_quality_labels,
    train_quality_model,
    save_model,
    load_model
)
from .analyzer import RealTimeAnalyzer

__all__ = [
    'parse_dataset',
    'extract_text_from_html',
    'FeatureExtractor',
    'detect_duplicates',
    'detect_thin_content',
    'create_quality_labels',
    'train_quality_model',
    'save_model',
    'load_model',
    'RealTimeAnalyzer'
]

