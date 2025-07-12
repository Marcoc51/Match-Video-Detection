"""
Middleware package for the API.
"""

from .cors import setup_cors

__all__ = ["setup_cors"] 