# CAP6415_F25_project-New-dataset-creation

This project provides a complete pipeline for preparing a custom dataset (LabelMe format) and training a YOLOv8 object detector.

## Features
- Rename and organize images
- Convert LabelMe JSON annotations to YOLO format
- Split dataset into train/val/test
- Train YOLOv8 model
- Evaluate and visualize results

## Getting Started

### 1. Clone the Repository
```
git clone https://github.com/zhuopuzhouComputerVision/CAP6415_F25_project-New-dataset-creation.git
cd CAP6415_F25_project-New-dataset-creation
```

### 2. Install Requirements

#### Prerequisites
- Python 3.8+
- [Anaconda](https://www.anaconda.com/) recommended
- NVIDIA GPU with updated drivers (for GPU training)

#### Option 1: Using Anaconda (Recommended)

1. Initialize conda for PowerShell (one-time setup):
```powershell
conda init powershell
```
**Important:** Close and reopen your PowerShell terminal after running this command.

2. Create a new conda environment:
```powershell
conda create -n yolo_project python=3.8
```

3. Activate the environment:
```powershell
conda activate yolo_project
```

4. Install PyTorch with CUDA support (for RTX 50-series or newer GPUs):
```powershell
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

For older GPUs, use CUDA 12.1:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

5. Install other dependencies:
```powershell
pip install ultralytics pillow
```

#### Option 2: Using pip only
```powershell
pip install ultralytics pillow
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### Verify Installation
```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### 3. Prepare Your Data
- Place your images in the `cat_image/` folder.
- Place your LabelMe JSON files in the `cat_label/` folder.

### 4. Run Dataset Preparation
```
python scripts/labelme_to_yolo.py
```
This will:
- Convert LabelMe JSONs to YOLO TXT
- Copy images to `data/images/{train,val,test}`
- Create YOLO label files in `data/labels/{train,val,test}`
- Generate `data/data.yaml`

### 5. Train YOLOv8
```
python train_yolo.py
```
- Training results and weights will be saved in `yolo_train/exp/`

### 6. Evaluate Model Performance
```
python eval_yolo.py
```
- Evaluation metrics and visualizations will be saved in `yolo_eval/exp/`

### 7. Test the Model on New Images
```
python infer_yolo.py
```
- Visualized predictions will be saved in `yolo_infer/exp/`

## Project Structure
```
cat_image/           # Raw images
cat_label/           # LabelMe JSON annotations
scripts/             # Data preparation scripts
train_yolo.py        # Training script
eval_yolo.py         # Evaluation script
infer_yolo.py        # Inference script
data/                # YOLO dataset (images, labels, data.yaml)
yolo_train/          # Training outputs
yolo_eval/           # Evaluation outputs
yolo_infer/          # Inference outputs
```

## Notes
- Adjust script parameters as needed for your dataset.
- See `conversation_summary_2025-11-06.txt` for a summary of the workflow and key concepts.


