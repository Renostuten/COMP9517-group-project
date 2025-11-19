# Deep Learning Pipeline for Insect Detection

This directory contains the deep learning implementations for the COMP9517 group project, focusing on insect detection and classification using state-of-the-art object detection models.

## 📋 Table of Contents

- [Overview](#overview)
- [Main Libraries](#main-libraries)
- [Files in this Directory](#files-in-this-directory)
- [YOLOv8 Training and Inference](#yolov8-training-and-inference)
- [Faster R-CNN Training and Evaluation](#faster-r-cnn-training-and-evaluation)
- [Results and Outputs](#results-and-outputs)

---

## Overview

This pipeline implements two deep learning approaches for insect detection:

1. **YOLOv8** - A modern, efficient single-stage detector optimized for real-time object detection
2. **Faster R-CNN** - A two-stage detector with region proposal network for high-accuracy detection

Both models are trained on the AgroPest-12 dataset containing 12 insect classes: Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs, Grasshoppers, Moths, Slugs, Snails, Wasps, and Weevils.

---

## Main Libraries

### Core Deep Learning Frameworks
- **PyTorch** (`torch`, `torchvision`) - Primary deep learning framework for model training and inference
- **Ultralytics** (`ultralytics`) - YOLOv8 implementation with easy-to-use training and prediction API

### Computer Vision
- **OpenCV** (`cv2`) - Image processing, loading, and visualization
- **PIL/Pillow** (`PIL`) - Image manipulation and format conversion
- **Albumentations** (`albumentations`) - Advanced data augmentation library

### Model Evaluation
- **scikit-learn** (`sklearn`) - Metrics computation (confusion matrix, precision-recall curves)
- **NumPy** (`numpy`) - Numerical operations and array manipulation
- **Pandas** (`pandas`) - Data organization and tabular display of results

### Visualization
- **Matplotlib** (`matplotlib.pyplot`) - Plotting graphs, curves, and images
- **Seaborn** (`seaborn`) - Statistical visualizations (heatmaps for confusion matrices)

---

## Files in this Directory

| File | Description |
|------|-------------|
| `train_yolo.ipynb` | YOLOv8 training and inference notebook |
| `train_faster_rcnn.ipynb` | Faster R-CNN training notebook |
| `faster_rcnn_evaluation.ipynb` | Comprehensive Faster R-CNN evaluation and metrics generation |
| `yolov8n.pt` | Pre-trained YOLOv8 nano weights (downloaded automatically) |
| `README.md` | This file - documentation for the pipeline |

---

## YOLOv8 Training and Inference

**Notebook:** `train_yolo.ipynb`

### Cell 1: Environment Setup and Verification

**Purpose:** Check Python environment, verify PyTorch installation, and confirm CUDA availability.

**What it does:**
- Imports required libraries
- Prints Python and PyTorch versions
- Checks if CUDA (GPU) is available
- Verifies data directory paths

**Expected Output:**
```
Python: 3.x.x
Torch: 2.x.x
CUDA available: True
CUDA device: [Your GPU Name]
Data root: ../data
```

**No parameters to modify** - this cell is purely diagnostic.

---

### Cell 2: Model Training

**Purpose:** Configure and train the YOLOv8 model on your dataset.

**What it does:**
- Loads pre-trained YOLOv8 nano model (`yolov8n.pt`)
- Configures training hyperparameters
- Initiates training with automatic validation
- Saves best model weights and training logs

**Key Parameters You Can Modify:**

| Parameter | Default | Description | Recommended Values |
|-----------|---------|-------------|-------------------|
| `data` | `'data/data.yaml'` | Path to dataset configuration | Keep as is |
| `epochs` | `50` | Number of training epochs | 25-100 (more = better but slower) |
| `imgsz` | `640` | Input image size (pixels) | 416, 640, 1280 |
| `batch` | `8` | Batch size per GPU | 4-16 (depends on GPU memory) |
| `device` | `0` or `'cpu'` | GPU device ID or CPU | `0` for GPU, `'cpu'` for CPU |
| `workers` | `4` | Number of data loading workers | 2-8 (depends on CPU cores) |
| `name` | `'yolov8_experiment'` | Experiment name for saving results | Any descriptive name |

**Model Size Options:**
- Change `YOLO('yolov8n.pt')` to:
  - `'yolov8s.pt'` - Small (better accuracy, slower)
  - `'yolov8m.pt'` - Medium (even better accuracy)
  - `'yolov8l.pt'` - Large (best accuracy, much slower)

**Training Output:**
Results are automatically saved to `../runs/detect/[name]/`:
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last epoch checkpoint
- Training curves, confusion matrix, and PR curves

---

### Cell 3: Model Inference and Evaluation

**Purpose:** Run predictions on test images and visualize results.

**What it does:**
1. Loads the trained model from saved weights
2. Allows you to select images by insect class name
3. Finds a random image containing your chosen insect
4. Runs prediction on the selected image
5. Displays detections in a pandas DataFrame
6. Shows annotated image with bounding boxes
7. Saves annotated image to results directory

**Key Parameters You Can Modify:**

| Parameter | Default | Description | How to Use |
|-----------|---------|-------------|------------|
| `INSECT_NAME` | `'Caterpillars'` | Class name to search for | Change to any class: `'Ants'`, `'Bees'`, `'Beetles'`, `'Caterpillars'`, `'Earthworms'`, `'Earwigs'`, `'Grasshoppers'`, `'Moths'`, `'Slugs'`, `'Snails'`, `'Wasps'`, `'Weevils'` |
| `conf` | `0.25` | Confidence threshold for predictions | 0.1-0.9 (lower = more detections, higher = fewer but more confident) |
| `imgsz` | `640` | Input image size for inference | Same as training size or 416, 640, 1280 |

**Model Path:**
- Default: `'../runs/detect/yolov8_experiment/weights/best.pt'`
- Change to `'yolov8n.pt'` for pre-trained model
- Or specify custom path to your trained weights

**Testing Different Images:**
- **By class name:** Set `INSECT_NAME = 'Ants'` to test on ant images
- **Random image:** Set `INSECT_NAME = ''` (empty string) to pick any test image
- **Specific image:** Replace the random selection with `sample_img = Path('../data/test/images/your_image.jpg')`

**Output Explanation:**
- **DataFrame columns:**
  - `x1, y1, x2, y2` - Bounding box coordinates (pixels)
  - `confidence` - Detection confidence score (0-1)
  - `class_id` - Numeric class identifier
  - `class_name` - Human-readable class name

**Results Location:**
- Annotated images: `../results/yolov8_predictions/[INSECT_NAME]/annotated_[image_name].jpg`

---

## Faster R-CNN Training and Evaluation

### Training Notebook: `train_faster_rcnn.ipynb`

**Purpose:** Train Faster R-CNN model with ResNet-50-FPN backbone.

#### Cell 1: Imports and Setup
**What it does:**
- Imports PyTorch, torchvision, and other required libraries
- Checks CUDA availability

**No parameters to modify** - diagnostic cell.

---

#### Cell 2: Dataset Class Definition
**What it does:**
- Defines custom `AgroPestDataset` class
- Converts YOLO format labels to Faster R-CNN format (xyxy coordinates)
- Applies image transformations and augmentations

**No direct parameters** - class definition only.

---

#### Cell 3: Configuration and Hyperparameters

**Purpose:** Set all training hyperparameters and paths.

**Key Parameters You Can Modify:**

| Parameter | Default | Description | Recommended Values |
|-----------|---------|-------------|-------------------|
| `TRAINING_MODE` | `'quick_test'` | Training mode selection | `'quick_test'` (500 images, 5 epochs) or `'full'` (all images, 12 epochs) |
| `BATCH_SIZE` | `4` | Batch size | 2-8 (limited by 8GB VRAM) |
| `NUM_WORKERS` | `6` | Data loading workers | 4-8 for i7-12700 |
| `LEARNING_RATE` | `0.001` | Initial learning rate | 0.0001-0.005 |
| `NUM_EPOCHS` | Auto-set | Number of epochs | Auto: 5 (quick) or 12 (full) |
| `GRADIENT_ACCUMULATION_STEPS` | `2` | Gradient accumulation | 2-4 (effective batch = BATCH_SIZE × this) |
| `USE_SUBSET` | Auto-set | Use training subset | Auto: True (quick) or False (full) |
| `IOU_THRESHOLD` | `0.5` | IoU threshold for NMS | 0.3-0.7 |
| `SCORE_THRESHOLD` | `0.05` | Minimum detection score | 0.01-0.5 |

**What it does:**
- Prints comprehensive configuration summary
- Shows hardware specs and estimated training time
- Validates all paths exist

**Training Time Estimates:**
- **Quick test mode:** ~1 hour
- **Full training:** ~5-6 hours (on RTX 3070 Ti)

---

#### Cell 4-6: Model Training
**What it does:**
- Initializes Faster R-CNN model with custom head
- Trains model with mixed precision (FP16)
- Saves checkpoints every 2 epochs
- Computes validation loss

**Parameters already configured in Cell 3** - just run these cells.

**Output:**
- Training/validation loss printed each epoch
- Checkpoints: `faster_rcnn_checkpoint_epoch_[N].pth`
- Best model: `best_faster_rcnn_model.pth`

---

### Evaluation Notebook: `faster_rcnn_evaluation.ipynb`

**Purpose:** Comprehensive model evaluation with multiple metrics and visualizations.

#### Cell 1: Imports
Standard library imports - no parameters.

---

#### Cell 2: Evaluation Configuration

**Key Parameters You Can Modify:**

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `CONFIDENCE_THRESHOLDS` | `np.arange(0.0, 1.01, 0.005)` | Confidence range for curves | Affects precision/recall/F1 curve granularity |
| `IOU_THRESHOLD` | `0.3` | IoU threshold for matching predictions to ground truth | 0.3-0.7 (higher = stricter matching) |

**What it does:**
- Sets up paths (automatically detected)
- Loads class names
- Configures device (GPU/CPU)

---

#### Cell 3: Model Loading
**What it does:**
- Loads trained model checkpoint
- Searches for `best_faster_rcnn_model.pth` or latest checkpoint
- Displays model info (epoch, training loss, validation loss)

**No parameters to modify** - automatic checkpoint detection.

---

#### Cell 4: Ground Truth Loading
**What it does:**
- Loads all test set annotations
- Converts YOLO format to model format
- Builds annotation dictionary

**Output:** Number of test images loaded with annotations.

---

#### Cell 5: Model Evaluation
**What it does:**
- Runs model on entire test set
- Collects all predictions with confidence scores
- Scales bounding boxes to original image sizes

**This may take several minutes** depending on test set size (~546 images).

---

#### Cell 6: Metrics Calculation

**Purpose:** Calculate precision, recall, F1 for each class.

**Parameter You Can Modify:**

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `conf_threshold` | `0.05` | Confidence threshold for metrics | Higher = fewer predictions, higher precision, lower recall |

**What it does:**
- Computes IoU between predictions and ground truth
- Calculates TP, FP, FN for each class
- Computes precision, recall, F1 scores
- Shows detailed debug output

---

#### Cell 7: Confidence Analysis
**What it does:**
- Analyzes model confidence scores across test images
- Shows score distributions
- Recommends optimal confidence threshold

**No parameters** - diagnostic cell.

---

#### Cell 8: Precision-Recall Curves
**What it does:**
- Generates PR curves for each class
- Computes Average Precision (AP) per class
- Saves plot as `BoxPR_curve.png`

**Output saved to:** `results/faster_rcnn_evaluation/BoxPR_curve.png`

---

#### Cell 9: F1-Confidence Curves
**What it does:**
- Shows how F1 score varies with confidence threshold
- Finds optimal confidence for best F1
- Saves plot as `BoxF1_curve.png`

---

#### Cell 10: Confusion Matrix
**What it does:**
- Generates confusion matrix (raw counts)
- Generates normalized confusion matrix (percentages)
- Shows which classes are commonly confused
- Saves both matrices as PNG files

**Output:**
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`

---

#### Cell 11: Recall-Confidence Curves
**What it does:**
- Shows recall vs confidence for all classes
- Individual curves for each insect class
- Saves plot as `BoxR_curve.png`

---

#### Cell 12: Precision-Confidence Curves
**What it does:**
- Shows precision vs confidence for all classes
- Individual curves for each insect class
- Saves plot as `BoxP_curve.png`

---

#### Cell 13: Comprehensive Summary
**What it does:**
- Calculates overall metrics (precision, recall, F1)
- Generates JSON summary with all results
- Saves `evaluation_summary.json`
- Prints final comparison-ready statistics

**Output:** All evaluation files ready for YOLO comparison in report.

---

## Results and Outputs

### YOLOv8 Results Location
```
../runs/detect/yolov8_experiment/
├── weights/
│   ├── best.pt              # Best model weights
│   └── last.pt              # Last epoch weights
├── results.png              # Training curves
├── confusion_matrix.png     # Confusion matrix
├── BoxPR_curve.png         # Precision-Recall curves
└── ...                     # Other evaluation plots

../results/yolov8_predictions/
└── [INSECT_NAME]/
    └── annotated_*.jpg     # Inference visualizations
```

### Faster R-CNN Results Location
```
Project root:
├── faster_rcnn_checkpoint_epoch_*.pth  # Training checkpoints
└── best_faster_rcnn_model.pth          # Best model

dl_pipeline/results/faster_rcnn_evaluation/
├── BoxPR_curve.png                     # Precision-Recall curves
├── BoxF1_curve.png                     # F1-Confidence curve
├── BoxR_curve.png                      # Recall-Confidence curve
├── BoxP_curve.png                      # Precision-Confidence curve
├── confusion_matrix.png                # Confusion matrix
├── confusion_matrix_normalized.png     # Normalized confusion matrix
└── evaluation_summary.json             # Complete metrics in JSON format
```

---
