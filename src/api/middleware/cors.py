"""
CORS middleware configuration for the football analysis API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List


def setup_cors(app: FastAPI, allowed_origins: List[str] = None) -> None:
    """
    Setup CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins. If None, uses default origins.
    """
    if allowed_origins is None:
        allowed_origins = [
            "http://localhost:3000",  # React development server
            "http://localhost:8080",  # Vue.js development server
            "http://localhost:4200",  # Angular development server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:4200",
            "*"  # Allow all origins in development (remove in production)
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
        ],
    ) 