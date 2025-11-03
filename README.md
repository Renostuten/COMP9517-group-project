# COMP9517 Group Project - Insect Detection

This repository contains two approaches for insect detection as part of the COMP9517 group project:

1. Deep Learning: YOLOv8-based object detection pipeline
2. Traditional ML: HOG + SVM classifier pipeline

Both methods are implemented to compare traditional machine learning approaches with modern deep learning techniques.

## Quick Start

1. Install dependencies:
```bash
# Install PyTorch with CUDA support (adjust URL for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

2. Validate your dataset (optional but recommended):
```bash
python3 scripts/validate_labels.py --data-dir data --split train
python3 scripts/validate_labels.py --data-dir data --split valid
python3 scripts/validate_labels.py --data-dir data --split test
```

Please note that the slug-282 training label files are empty by default so the validation script will flag those files as empty.

3. Open and run the training notebook:
```bash
code dl_pipeline/train_yolo.ipynb
```

## Project Structure

```
.
├── data/                    # Dataset directory
│   ├── data.yaml           # Dataset configuration for YOLOv8
│   ├── train/              # Training data
│   │   ├── images/         
│   │   └── labels/         # YOLO format labels
│   ├── valid/              # Validation data
│   └── test/               # Test data
├── dl_pipeline/            # Deep Learning (YOLOv8) implementation
│   └── train_yolo.ipynb    # YOLOv8 training notebook
├── ml_pipeline/            # Traditional ML implementation
│   └── hog_svm_train.ipynb # HOG+SVM training notebook
├── results/                # Training outputs and comparisons
└── scripts/
    └── validate_labels.py  # Dataset validation utility
```

## Implementation Approaches

### 1. Deep Learning (YOLOv8)
- Modern object detection using YOLOv8
- End-to-end training pipeline
- Real-time detection capabilities
- See sections below for detailed YOLOv8 setup and training

### 2. Traditional ML (HOG + SVM)
- Classical computer vision approach
- Histogram of Oriented Gradients (HOG) for feature extraction
- Support Vector Machine (SVM) for classification
- Implementation details in `ml_pipeline/hog_svm_train.ipynb`
- Note: This pipeline is under development

## Dataset Format

The dataset follows the YOLO format:

- Images: JPG/PNG in `data/{split}/images/`
- Labels: Text files in `data/{split}/labels/`
- Label format: One object per line: `class_id x_center y_center width height`
  - All values are normalized [0-1]
  - (x_center, y_center) is the box center
  - (width, height) are box dimensions

Example label line:
```
0 0.716797 0.395833 0.216406 0.147222
```

## Training Tips

1. Check your GPU and CUDA:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

2. Common training parameters (adjust in notebook):
```python
model.train(
    data='data/data.yaml',
    epochs=100,          # Total epochs
    imgsz=640,          # Image size
    batch=16,           # Batch size (adjust for your GPU)
    device=0,           # GPU index (or 'cpu')
    workers=4,          # DataLoader workers
    patience=50,        # Early stopping patience
)
```

3. Monitor training:
- Metrics are saved to `runs/detect/train*/`
- Use TensorBoard or WandB for visualization
- Check `results.csv` for per-epoch metrics

## Troubleshooting

1. CUDA/GPU issues:
   - Verify CUDA toolkit matches PyTorch version
   - Try reducing batch size if OOM
   - Use `device='cpu'` for CPU-only training

2. Dataset issues:
   - Run validation script to check labels
   - Ensure all paths in data.yaml are correct
   - Check class IDs match data.yaml

3. Common errors:
   - "CUDA out of memory": Reduce batch size
   - "No labels found": Check data.yaml paths
   - "Invalid label format": Run validator

## Performance Comparison

A detailed performance comparison between YOLOv8 and HOG+SVM approaches will be added here after both implementations are complete. Metrics will include:

- Detection accuracy
- Processing speed
- Resource usage
- Ease of deployment

## Contributing

1. Create a feature branch
2. Make changes
3. Validate dataset if you modified data
4. Test training for at least 1 epoch (YOLOv8) or run validation (HOG+SVM)
5. Create pull request

## Method Selection Guide

- **YOLOv8**: Choose when you need:
  - Real-time detection
  - High accuracy
  - GPU availability
  - Modern deep learning approach

- **HOG+SVM**: Consider when you have:
  - Limited computational resources
  - Need for model interpretability
  - Smaller dataset
  - CPU-only deployment requirements