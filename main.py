#!/usr/bin/env python3
"""
Main entry point for the Match Video Detection application.
This script provides a command-line interface to run the football analysis pipeline.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.main import main

if __name__ == "__main__":
    main()