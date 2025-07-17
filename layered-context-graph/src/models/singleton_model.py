#!/usr/bin/env python3
"""
Singleton Model Loader
======================
This module provides a singleton loader for the QwQModel to ensure that
the model is loaded only once and persists between runs for faster debugging.
"""

import logging
from .qwq_model import QwQModel

logger = logging.getLogger(__name__)

_cached_model = None

def get_singleton_model(model_path: str, device=None):
    """
    Loads the QwQModel if it hasn't been loaded yet, otherwise returns
    the cached instance.
    """
    global _cached_model
    if _cached_model is None:
        logger.info("Loading model for the first time...")
        _cached_model = QwQModel(model_path, device)
    else:
        logger.info("Returning cached model instance.")
    return _cached_model

def clear_singleton_model():
    """Clears the cached model instance."""
    global _cached_model
    if _cached_model is not None:
        logger.info("Clearing cached model instance.")
        # Potentially add more complex cleanup here if needed (e.g., del model)
        _cached_model = None
