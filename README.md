# EfficientNet-RetinaNet Object Detection

A PyTorch implementation of object detection using EfficientNet backbone with RetinaNet detection head for urine sediment analysis.

## 📋 Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [File Structure](#file-structure)

## 🔍 Overview

This project implements an object detection model that combines:
- **EfficientNet** (B0-B5) as feature extraction backbone
- **RetinaNet** as single-stage object detection head
- **Feature Pyramid Network (FPN)** for multi-scale detection
- **Focal Loss** for addressing class imbalance

The model is trained on urine sediment dataset for medical object detection tasks.

## 🏗️ Model Architecture

### EfficientNet Backbone
- **MBConvBlock**: Mobile Inverted Residual with Squeeze-and-Excitation
- **Multi-scale feature extraction**: Outputs C3, C4, C5 feature maps
- **Efficient scaling**: Compound scaling of depth, width, and resolution

### RetinaNet Detection Head
- **Feature Pyramid Network**: Creates P3-P7 pyramid levels
- **Classification Head**: 4-layer CNN for class prediction
- **Regression Head**: 4-layer CNN for bounding box regression
- **Anchor Generation**: Multi-scale and multi-aspect ratio anchors

## 📊 Dataset

### Format
- **Input**: COCO format annotations converted to CSV
- **Structure**: 
  ```
  image_path,xmin,ymin,xmax,ymax,label
  ```

### Required Files
- `train/_annotations.csv` - Training annotations
- `valid/_annotations.csv` - Validation annotations  
- `test/_annotations.csv` - Test annotations
- `classes.csv` - Class label mappings

## 🚀 Installation

### Requirements
```bash
python version > 3.10
```

### Clone Repository
```bash
git clone https://github.com/dikaizm/efficientnet-retinanet-object-detection.git
cd efficientnet-retinanet-object-detection
```

### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install requirements.txt
```

## ⚡ Quick Start

### 1. Prepare Dataset
```python
# Download and extract dataset (modify file_id as needed)
import gdown
import zipfile

file_id = "YOUR_DATASET_FILE_ID"
gdown.download(f"https://drive.google.com/uc?id={file_id}", "dataset.zip")

with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")
```

### 2. Convert COCO to CSV
```python
from notebooks.efficientnet_retinanet_model import coco_to_csv

# Convert annotations for each split
for split in ["train", "valid", "test"]:
    coco_to_csv(
        coco_json_path=f"data/{split}/_annotations.coco.json",
        images_dir=f"data/{split}",
        output_csv_path=f"data/{split}/_annotations.csv"
    )
```

### 3. Train Model
```python
# Set training parameters
train_config = {
    "train_csv": "data/train/_annotations.csv",
    "test_csv": "data/valid/_annotations.csv", 
    "labels_csv": "data/classes.csv",
    "model_type": "b4",  # EfficientNet variant
    "epochs": 25,
    "batch_size": 4
}

# Start training
history = train(**train_config)
```

### 4. Run Inference
```python
# Load trained model and run inference
detections, output_image = run_inference(
    model_path="runs/best_model.pt",
    image_path="path/to/test/image.jpg",
    labels_csv="data/classes.csv",
    output_path="results/inference_output.png",
    confidence_threshold=0.5
)
```

## 🎯 Training

### Configuration
Key training parameters in the notebook:

```python
# Training Configuration
model_type = "b4"           # EfficientNet variant: b0, b1, b2, b3, b4, b5
epochs = 25                 # Number of training epochs
batch_size = 4              # Batch size (adjust based on GPU memory)
learning_rate = 1e-5        # Adam optimizer learning rate
```

### Training Process
1. **Data Loading**: CSV dataset with augmentation (rotation, flipping, etc.)
2. **Model Initialization**: EfficientNet-B4 backbone + RetinaNet head
3. **Optimization**: Adam optimizer with ReduceLROnPlateau scheduler
4. **Loss Functions**: 
   - Focal Loss for classification
   - Smooth L1 Loss for regression
5. **Evaluation**: mAP calculation after each epoch
6. **Checkpointing**: Saves best model based on validation mAP

### Training Outputs
- `checkpoints/` - Epoch-wise model checkpoints
- `runs/best_model.pt` - Best performing model
- `runs/training_history.csv` - Loss and mAP metrics
- `runs/training_curves.png` - Training visualization

## 🔮 Inference

### Inference Pipeline
1. **Preprocessing**: Image normalization and resizing
2. **Forward Pass**: Model prediction (classification + regression)
3. **Post-processing**: 
   - Apply regression deltas to anchors
   - Filter by confidence threshold
   - Non-Maximum Suppression (NMS)
4. **Visualization**: Draw bounding boxes with class labels

### Inference Function
```python
detections = run_inference(
    model_path="runs/best_model.pt",
    image_path="test_image.jpg", 
    labels_csv="data/classes.csv",
    output_path="output.png",
    confidence_threshold=0.5
)
```

### Output Format
```python
# Detection format
{
    'label': 'class_name',
    'confidence': 0.85,
    'bbox': [x1, y1, x2, y2]
}
```

## 📈 Results

### Training Metrics
- **Classification Loss**: Focal loss for handling class imbalance
- **Regression Loss**: Smooth L1 loss for bounding box regression  
- **Validation mAP**: Mean Average Precision on validation set

### Model Performance
- Supports multiple EfficientNet variants (B0-B5)
- Real-time inference capability
- High accuracy on medical imaging tasks

## 📁 File Structure

```
efficientnet-retinanet-object-detection/
├── model/
│   ├── retinanet.py           # Main RetinaNet implementation
│   ├── efficientnet_model.py  # EfficientNet backbone
│   ├── efficientnet_utils.py  # EfficientNet configurations
│   ├── anchors.py             # Anchor generation
│   ├── losses.py              # Loss functions
│   ├── model_dataloader.py    # Dataset and transforms
│   ├── eval.py                # Evaluation metrics
│   └── utils.py               # Utility functions
├── notebooks/
│   └── efficientnet-retinanet-model.ipynb  # Main training notebook
├── data/                      # Dataset directory
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── classes.csv
├── runs/                      # Training outputs
│   ├── best_model.pt
│   ├── training_history.csv
│   └── training_curves.png
├── checkpoints/               # Model checkpoints
└── requirements.txt
```

## 🔄 Reproduction Steps

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   - Download dataset using provided Google Drive links
   - Convert COCO annotations to CSV format
   - Verify class labels in `classes.csv`

3. **Training**:
   - Open `notebooks/efficientnet-retinanet-model.ipynb`
   - Execute cells sequentially
   - Monitor training progress and validation mAP

4. **Evaluation**:
   - Use trained model for inference
   - Compare predictions with ground truth
   - Calculate final performance metrics

## 📝 Notes

- **GPU Memory**: Reduce batch size if encountering OOM errors
- **Model Variants**: Larger EfficientNet variants (B4, B5) provide better accuracy but require more memory
- **Confidence Threshold**: Adjust based on precision/recall requirements
- **Training Time**: Approximately 2-3 minutes per epoch on modern GPUs

## 🏆 Competition Submission

For competition organizers to reproduce results:

1. Follow installation and setup steps
2. Use provided dataset and configuration
3. Execute the complete notebook from start to finish
4. Final model and metrics will be saved in `runs/` directory
5. Inference results can be generated using the trained model

The complete training pipeline, evaluation metrics, and model weights are provided for full reproducibility.