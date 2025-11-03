#!/usr/bin/env python3
"""
Validate YOLO format labels in the dataset.
Checks:
1. Label format (class x_center y_center width height)
2. Normalized coordinates [0-1]
3. Image-label pairs exist
4. No empty labels (unless intentional)
5. Class IDs are valid
"""

import os
import glob
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import cv2
import numpy as np
from tqdm import tqdm

def find_image_label_pairs(data_dir: str, split: str = "train") -> Tuple[List[str], List[str]]:
    """Find all image-label pairs in the dataset."""
    img_dir = Path(data_dir) / split / "images"
    label_dir = Path(data_dir) / split / "labels"
    
    # Get all image files
    image_files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files.extend(list(img_dir.glob(f"*{ext}")))
    
    # Get corresponding label files
    label_files = []
    orphaned_labels = []
    missing_labels = []
    
    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            label_files.append(label_path)
        else:
            missing_labels.append(img_path)
    
    # Check for labels without images
    for label_path in label_dir.glob("*.txt"):
        img_found = False
        for ext in [".jpg", ".jpeg", ".png"]:
            if (img_dir / f"{label_path.stem}{ext}").exists():
                img_found = True
                break
        if not img_found:
            orphaned_labels.append(label_path)
    
    return image_files, label_files, missing_labels, orphaned_labels

def validate_label_content(label_path: str, num_classes: int) -> List[str]:
    """Validate single label file format and content."""
    errors = []
    
    try:
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
            
        if not lines:
            return [f"Warning: Empty label file {label_path}"]
            
        for i, line in enumerate(lines, 1):
            try:
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"Line {i}: Expected 5 values, got {len(parts)}")
                    continue
                
                # Parse values
                cls = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # Validate class ID
                if cls < 0 or (num_classes > 0 and cls >= num_classes):
                    errors.append(f"Line {i}: Invalid class ID {cls}")
                
                # Validate coordinates
                if any(c < 0.0 or c > 1.0 for c in coords):
                    errors.append(f"Line {i}: Coordinates must be normalized [0-1], got {coords}")
                
                # Basic sanity checks
                if coords[2] <= 0 or coords[3] <= 0:
                    errors.append(f"Line {i}: Width/height must be positive, got w={coords[2]}, h={coords[3]}")
                
            except ValueError as e:
                errors.append(f"Line {i}: Failed to parse values: {e}")
                
    except Exception as e:
        errors.append(f"Failed to read file: {e}")
        
    return errors

def main():
    parser = argparse.ArgumentParser(description="Validate YOLO dataset format")
    parser.add_argument("--data-dir", type=str, default="data",
                      help="Path to data directory containing train/valid/test splits")
    parser.add_argument("--split", type=str, default="train",
                      help="Dataset split to validate (train/valid/test)")
    parser.add_argument("--num-classes", type=int, default=-1,
                      help="Number of classes (-1 to skip class ID validation)")
    args = parser.parse_args()
    
    print(f"Validating {args.split} split in {args.data_dir}...")
    
    # Find all image-label pairs
    image_files, label_files, missing_labels, orphaned_labels = find_image_label_pairs(
        args.data_dir, args.split
    )
    
    print(f"\nFound:")
    print(f"- {len(image_files)} images")
    print(f"- {len(label_files)} labels")
    print(f"- {len(missing_labels)} images without labels")
    print(f"- {len(orphaned_labels)} labels without images")
    
    if missing_labels:
        print("\nImages missing labels:")
        for path in missing_labels[:10]:  # Show first 10
            print(f"- {path}")
        if len(missing_labels) > 10:
            print(f"... and {len(missing_labels)-10} more")
            
    if orphaned_labels:
        print("\nOrphaned label files:")
        for path in orphaned_labels[:10]:
            print(f"- {path}")
        if len(orphaned_labels) > 10:
            print(f"... and {len(orphaned_labels)-10} more")
    
    # Validate label content
    print("\nValidating label contents...")
    all_errors: Dict[str, List[str]] = {}
    
    for label_path in tqdm(label_files):
        errors = validate_label_content(str(label_path), args.num_classes)
        if errors:
            all_errors[str(label_path)] = errors
    
    if all_errors:
        print("\nLabel format errors found:")
        for path, errors in list(all_errors.items())[:10]:  # Show first 10 files with errors
            print(f"\n{path}:")
            for err in errors:
                print(f"  - {err}")
        if len(all_errors) > 10:
            print(f"\n... and {len(all_errors)-10} more files with errors")
    else:
        print("\nNo label format errors found!")
        
if __name__ == "__main__":
    main()