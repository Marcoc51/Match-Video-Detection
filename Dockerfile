# Dockerfile for YOLOv8 with Python 3.10 and CUDA 12.2
# Based on NVIDIA's CUDA image for Ubuntu 22.04
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
# This includes Python 3.10, pip, and other necessary libraries
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip python-is-python3 \
    ffmpeg libsm6 libxext6 git curl wget unzip nano && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Python packages
RUN pip install --upgrade pip
RUN pip install ultralytics opencv-python jupyter matplotlib numpy pandas

# Disable YOLOv8 auto-update feature
# This prevents the YOLOv8 package from checking for updates automatically
ENV YOLOv8_AUTOUPDATE=false
