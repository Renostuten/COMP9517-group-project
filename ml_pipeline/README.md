# Traditional Computer Vision Pipeline for Insect Detection

This directory contains the traditional machine learning implementation for the insect detection project. This notebook (`[ML]Detector+Classificaiton.ipynb`) implements a complete pipeline from feature extraction to evaluation without using deep neural networks.

## Table of Contents

- [Overview](#overview)
- [Main Libraries](#main-libraries)
- [Notebook Structure & Execution](#notebook-structure--execution)
    - [Cell 1: Setup & Configuration](#cell-1-environment-setup-and-configuration)
    - [Cell 2: Utilities](#cell-2-utilities)
    - [Cell 3: Classification Module](#cell-3-classification-module)
    - [Cell 4: Detector Module](#cell-4-detector-module)
    - [Cell 5: Main Execution (Training)](#cell-5-main-execution-training)
    - [Cell 6: Evaluation & Visualisation](#cell-6-evaluation--visualisation)
- [Results Interpretation](#results-interpretation)

---

## Overview

This pipeline combines classical computer vision techniques to detect and classify 12 insect species. Unlike deep learning approaches (YOLO/Faster R-CNN), this method relies on handcrafted features and statistical learning.

**System Architecture:**
1.  **Region Proposal**: Generates candidate boxes using **Background Subtraction** (Mahalanobis distance) & Morphological operations to find "blobs" likely to be insects.
2.  **Binary Detector**: Filters these candidates (Insect vs. Background) using **HOG features** trained on a **Linear SVM**.
3.  **Species Classifier**: Classifies the validated regions into 12 species using a fusion of **SIFT**, **HOG**, and **Color Histograms**, trained on a non-linear **SVM (RBF Kernel)**.

---

## Main Libraries

* **OpenCV** (`cv2`): Image processing, SIFT feature extraction, Background subtraction.
* **Scikit-Learn** (`sklearn`): K-Means (Visual Words), SVM (SVC, LinearSVC), Scalers, Pipeline.
* **Scikit-Image** (`skimage`): HOG feature extraction implementation.
* **Matplotlib**: Visualization of bounding boxes and results.
* **Tqdm**: Progress bar visualization.

---

## Notebook Structure & Execution

The notebook is designed to be run sequentially. Below is a detailed guide for each cell block.

### Cell 1: Environment Setup and Configuration

**Purpose:** Imports libraries and defines global hyperparameters for both the classifier and detector.

**Key Parameters You Can Modify:**

| Dictionary | Parameter | Default | Description |
|:---|:---|:---|:---|
| `CLF_CONFIG` | `K_VOCAB_SIZE` | `800` | Number of clusters for BoVW. Higher = more detailed texture features. |
| `CLF_CONFIG` | `SIFT_STEP` | `16` | Step size for dense SIFT. Lower (e.g., 8) = finer detail but slower. |
| `CLF_CONFIG` | `C` | `2.0` | SVM Regularization. Higher = stricter margin (risk of overfitting). |
| `CLF_CONFIG` | `W_BOVW` | `1.0` | Weight importance for SIFT texture features. |
| `CLF_CONFIG` | `W_COLOR` | `1.0` | Weight importance for Color Histogram features. |
| `DETECTOR` | `DET_SCORE_THRESH` | `0.5` | Confidence threshold for the binary detector to accept a box. |
| `SAMPLING` | `TRAIN_MAX_IMAGES` | `11000` | Max images to use for training (Lower this if memory is full). |
| `SAMPLING` | `MAX_POS_PER_IMG` | `50` | Max insect crops to extract per image during training. |

**Path Setup:**
Ensure `DATA_DIR` points to your dataset root. The expected structure is:
```text
../data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
### Cell 2: Utilities

**Purpose:** Helper functions for file handling and geometry.

**Key Functions:**
* `read_yolo_boxes`: Parses YOLO format text files (`class_id center_x center_y w h`) into pixel coordinates.
* `iou_xywh`: Calculates Intersection over Union between two boxes.
* `nms_xywh`: Performs Non-Maximum Suppression to merge overlapping boxes based on confidence scores.

**Action:** Just run this cell. No parameters to modify.

---

### Cell 3: Classification Module

**Purpose:** Defines the logic for the 12-class species classifier.

**Feature Extraction Logic:**
1.  **Dense SIFT**: Extracts local texture descriptors.
2.  **BoVW (Bag of Visual Words)**: Maps SIFT descriptors to the nearest vocabulary cluster.
3.  **HOG**: Captures global shape information.
4.  **Color Histograms**: Captures color distribution in LAB space.
5.  **Feature Fusion**: Concatenates all features into a single vector.

**Training Logic (`load_classifier_data_and_train`):**
* Loads cropped insect images.
* Trains **MiniBatchKMeans** to create the Visual Vocabulary.
* Trains a **SVC (Support Vector Classifier)** with an RBF kernel.

**Action:** Run this cell to define the functions. The actual training happens in Cell 5.

---

### Cell 4: Detector Module

**Purpose:** Defines the logic for the Binary (Insect vs. Background) detector and Region Proposals.

**Key Components:**
* **Region Proposal**: `make_fg_mask_simple` uses statistical background modeling (Mahalanobis distance on LAB color) to find foreground regions.
* **HOG Feature**: `extract_hog_feature` computes shape descriptors specifically for the binary classifier.
* **Dataset Builder**: `build_binary_dataset_for_split` creates a balanced dataset of Insect crops (Positive) and Random Background crops (Negative).
* **Detection Logic**: `detect_in_image` pipelines the whole process:
    * Input Image -> Region Proposal -> HOG Feature -> SVM Score -> NMS -> Final Boxes.

**Action:** Run this cell to define the detector logic.

---

### Cell 5: Main Execution (Training)

**Purpose:** Triggers the actual training process for both models.

**What happens when you run this:**
1.  **Classifier Training**:
    * Loads training crops from the dataset.
    * Trains the K-Means vocabulary.
    * Trains the Multi-class SVM.
2.  **Detector Training**:
    * Builds a dataset of Positives (Insects) and Negatives (Background).
    * Trains the **LinearSVC** for fast binary classification.


**Expected Console Output:**
```
 Training Multi Class Classifier

Classifier Loading Training Data
Loading Train Crops: 100%|██████████| 11000/11000 [00:14<00:00, 765.43it/s]
Loaded 11000 training crops.
Training BoVW Voca
Training SVC

Training Binary Detector
Build binary HOG dataset @ train: 100%|██████████| 11000/11000 [00:45<00:00, 241.54it/s]
[INFO] train split: total sample = 33000
[INFO]   - Positive(Insect=1): 11000
[INFO]   - Negative(Background=0): 22000
Build binary HOG dataset @ valid: 100%|██████████| 1000/1000 [00:04<00:00, 238.51it/s]
[INFO] valid split: total sample = 3000
[INFO]   - Positive(Insect=1): 1000
[INFO]   - Negative(Background=0): 2000

Training Detector SVM
```
### Cell 6: Evaluation & Visualisation

**Purpose:** Evaluates the full pipeline on the **Test Set** and visualizes the detection results.

**Evaluation Process (`evaluate_full_system`):**
1.  **Load Data**: Iterates through all images in the Test set.
2.  **Detection**: Runs the **Region Proposal + Binary Detector** to find potential insect boxes.
3.  **Classification**: Passes the detected boxes to the **Species Classifier** to predict the insect type.
4.  **Matching**: Compares predictions with Ground Truth boxes using an IoU threshold of 0.5.
5.  **Metrics**: Computes **mAP (Mean Average Precision)** and per-class **Precision, Recall, and F1-Score**.

**Expected Console Output:**

1.  **Progress Bar**: A `tqdm` progress bar showing the evaluation status across all test images.
2.  **Performance Table**: A detailed text table displaying the following metrics for **each of the 12 insect classes**:
    * **Precision, Recall, F1-Score**: Classification metrics.
    * **TP (True Positive)**: Count of correctly detected insects.
    * **FP (False Positive)**: Count of incorrect detections.
    * **GT (Ground Truth)**: Total number of actual insects in the test set.
3.  **Final Scores**:
    * **Macro Average**: Average performance across all classes.
    * **mAP@0.50**: The final Mean Average Precision score at IoU 0.5.
4.  **Visualizations (Plots)**:
    * The code will display random test images with bounding boxes overlaid.
    * **Green Boxes**: Represent correct predictions (Correct location & Correct species).
    * **Red Boxes**: Represent errors (Background detected as insect or Wrong species prediction).

**Visualization (`visualize_detections`):**
* Displays random samples of the model's performance.
* **Green Box (True Positive)**: Correctly detected the insect AND correctly classified the species.
* **Red Box (False Positive/Misclassification)**: Detected background as an insect OR classified the insect as the wrong species.

**Parameters You Can Modify:**
```python
visualize_detections(
    detection_records,
    num_good=3,    # Number of successful examples (Green boxes) to show
    num_bad=3,     # Number of failed examples (Red boxes) to show
    score_thr=0.3  # Minimum confidence score to consider a valid detection
)
