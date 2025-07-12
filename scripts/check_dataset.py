#!/usr/bin/env python3
"""
Script to check dataset preparation results and verify annotations.
"""

import os
from pathlib import Path
import yaml

def check_dataset():
    """Check the prepared dataset for training."""
    
    # Check if training outputs directory exists
    output_dir = Path("training_outputs")
    if not output_dir.exists():
        print("❌ training_outputs directory not found!")
        print("Run data preparation first: python train_cross_detection.py --prepare-data-only")
        return
    
    print("🔍 Checking dataset preparation results...")
    print("="*50)
    
    # Check directory structure
    required_dirs = [
        "images/train",
        "images/val", 
        "labels/train",
        "labels/val"
    ]
    
    for dir_path in required_dirs:
        full_path = output_dir / dir_path
        if full_path.exists():
            file_count = len(list(full_path.glob("*")))
            print(f"✅ {dir_path}: {file_count} files")
        else:
            print(f"❌ {dir_path}: Directory not found")
    
    # Check dataset.yaml
    dataset_yaml = output_dir / "dataset.yaml"
    if dataset_yaml.exists():
        print(f"✅ dataset.yaml: Found")
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   - Classes: {config.get('nc', 'N/A')}")
        print(f"   - Class names: {config.get('names', 'N/A')}")
    else:
        print("❌ dataset.yaml: Not found")
    
    # Count annotations
    train_labels = list((output_dir / "labels/train").glob("*.txt"))
    val_labels = list((output_dir / "labels/val").glob("*.txt"))
    
    train_annotations = 0
    val_annotations = 0
    
    # Count annotations in training set
    for label_file in train_labels:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            train_annotations += len(lines)
    
    # Count annotations in validation set
    for label_file in val_labels:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            val_annotations += len(lines)
    
    print(f"\n📊 ANNOTATION STATISTICS:")
    print(f"Training annotations: {train_annotations}")
    print(f"Validation annotations: {val_annotations}")
    print(f"Total annotations: {train_annotations + val_annotations}")
    
    # Check for empty label files
    empty_train_files = 0
    empty_val_files = 0
    
    for label_file in train_labels:
        with open(label_file, 'r') as f:
            if not f.read().strip():
                empty_train_files += 1
    
    for label_file in val_labels:
        with open(label_file, 'r') as f:
            if not f.read().strip():
                empty_val_files += 1
    
    print(f"\n📁 EMPTY LABEL FILES:")
    print(f"Training: {empty_train_files} empty files")
    print(f"Validation: {empty_val_files} empty files")
    
    # Overall assessment
    print("\n" + "="*50)
    if train_annotations + val_annotations == 0:
        print("❌ CRITICAL: No annotations found!")
        print("The training will fail. Please check the data preparation logic.")
        print("Consider:")
        print("1. Adjusting cross detection criteria")
        print("2. Checking if videos contain crosses")
        print("3. Verifying ball detection is working")
    elif train_annotations + val_annotations < 10:
        print("⚠️  WARNING: Very few annotations found!")
        print("Training may not be effective with so few examples.")
    else:
        print("✅ Dataset looks good for training!")
        print(f"Found {train_annotations + val_annotations} cross annotations")
    
    # Show sample annotations
    if train_annotations > 0:
        print(f"\n📝 SAMPLE ANNOTATIONS:")
        sample_file = train_labels[0]
        print(f"File: {sample_file.name}")
        with open(sample_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:3]):  # Show first 3 annotations
                print(f"  {i+1}: {line.strip()}")
            if len(lines) > 3:
                print(f"  ... and {len(lines) - 3} more")

if __name__ == "__main__":
    check_dataset() 