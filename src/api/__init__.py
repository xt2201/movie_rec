"""
Flask API module for movie recommendation service.
"""
from .app import create_app
from .inference import InferenceService

__all__ = [
    "create_app",
    "InferenceService",
]
