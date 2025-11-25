
# Salient Object Detection - README

## Environment Setup

This project requires Python 3.10.19 due to TensorFlow compatibility. Since this version is no longer supported directly from the Python website, an **Anaconda environment** was used to manage dependencies and versions easily.

To set up the environment using Anaconda:

```bash
conda create -n sod_env python=3.10.9
conda activate sod_env
pip install tensorflow==2.12.0
pip install numpy==1.23.5
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install ipywidgets
```


## Checkpoint

This project includes a dedicated checkpoint folder that stores a pre-trained version of the model. Its purpose is to enable quick evaluation, demonstration, or continued training without requiring a full retraining cycle. If you wish to train the model from scratch, you must delete the entire checkpoint folder before initiating the training process; otherwise, the training script will automatically load the latest saved state and resume from it. During training, new checkpoints are generated to record the model’s progress. This mechanism helps prevent data loss in the event of an interruption, allowing training to continue seamlessly upon restart. To avoid unintended overfitting caused by repeatedly resuming from previous runs, ensure that the checkpoint folder is removed before starting any fresh training session.

## Dataset

The dataset used is DUTS, containing two folders:
```
Dataset/
├── Image/
└── Mask/
```

- `Image/` contains images.
- `Mask/` contains binary ground truth saliency masks.

If you wish to use your own dataset, **replace the contents** of `Image/` and `Mask/` with your own, but **retain the exact folder structure and file naming conventions** for compatibility with the existing pipeline.

## How to Run the Program

### 1. `data_loader.py`
Handles all data loading, preprocessing, and augmentation. No direct execution required.

### 2. `sod_model.py`
Defines the architecture and loss functions. Used internally by other scripts.

### 3. `train.py`
Trains the model from scratch using the training and validation sets.

```bash
python train.py
```

This will start training for 25 epochs and save the best model checkpoint automatically.

### 4. `evaluate.py`
Evaluates the best model saved from training and shows key metrics and saliency visualizations.

```bash
python evaluate.py
```

### 5. `app.ipynb`
A Jupyter Notebook interface that allows you to upload an image, perform inference, and view the predicted saliency mask along with inference time.

---
Ensure the `Dataset` folder is located in the same root directory as the project files for proper path resolution.
