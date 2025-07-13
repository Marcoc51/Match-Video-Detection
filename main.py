#!/usr/bin/env python3
"""
Main entry point for the Match Video Detection application.
This script provides a command-line interface to run the football analysis pipeline.
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Match Analysis")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--passes", action="store_true", help="Enable pass detection")
    parser.add_argument("--possession", action="store_true", help="Enable possession tracking")
    parser.add_argument("--crosses", action="store_true", help="Enable cross detection")
    
    args = parser.parse_args()
    
    result = main(
        video_path=args.video_path,
        project_root=Path(args.project_root),
        passes=args.passes,
        possession=args.possession,
        crosses=args.crosses
    )
    
    print("Analysis completed successfully!")