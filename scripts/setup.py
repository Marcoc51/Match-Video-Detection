#!/usr/bin/env python3
"""
Setup script for the Match Video Detection project.
This script helps initialize the project structure and dependencies.
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install project dependencies."""
    print("Installing project dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    print("Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/training/images",
        "data/training/labels",
        "data/validation/images",
        "data/validation/labels",
        "models/yolo",
        "models/trained",
        "models/deployed",
        "outputs/videos",
        "outputs/results",
        "outputs/logs",
        "mlflow/experiments"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("✅ All directories created successfully!")

def check_models():
    """Check if required models exist."""
    print("Checking for required models...")
    
    model_path = Path("models/yolo/best.pt")
    if not model_path.exists():
        print("⚠️  Warning: YOLO model not found at models/yolo/best.pt")
        print("   Please download or train a YOLO model and place it in models/yolo/")
    else:
        print("✅ YOLO model found!")

def main():
    """Main setup function."""
    print("🚀 Setting up Match Video Detection project...")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Check models
    check_models()
    
    # Install dependencies
    if install_dependencies():
        print("\n🎉 Project setup completed successfully!")
        print("\nNext steps:")
        print("1. Place your video files in data/raw/")
        print("2. Download or train a YOLO model and place it in models/yolo/")
        print("3. Run: python main.py --help")
    else:
        print("\n❌ Project setup failed. Please check the errors above.")

if __name__ == "__main__":
    main() 