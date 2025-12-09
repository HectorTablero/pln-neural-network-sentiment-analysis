"""
Corpus Management System for BoardGameGeek Reviews

A modular, professional system for managing and analyzing game review data
for sentiment analysis tasks.
"""

from .reader import CorpusReader
from .document import Document
from .corpus import Corpus
from .preprocessing import PreprocessingPipeline
from .linguistic_analyzer import LinguisticAnalyzer
from .feature_extractor import FeatureExtractor
from .vector_manager import VectorManager
from .persistence import PersistenceManager

__version__ = "1.0.0"
__all__ = [
    "CorpusReader",
    "Document",
    "Corpus",
    "PreprocessingPipeline",
    "LinguisticAnalyzer",
    "FeatureExtractor",
    "VectorManager",
    "PersistenceManager",
]
