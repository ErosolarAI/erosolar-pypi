"""
Erosolar - AI-powered chat interface with DeepSeek
"""

__version__ = "1.0.0"
__author__ = "ErosolarAI"
__description__ = "AI-powered chat interface with DeepSeek, featuring tool integrations and embeddings"

from .app import app, get_user_data_dir, DATABASE_PATH

__all__ = ["app", "get_user_data_dir", "DATABASE_PATH"]
