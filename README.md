[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/uOK6onxy)
# ML Assignment 3: Medical Image Segmentation

This repository contains the implementation for Assignment 3 of the Machine Learning (CSIE5043) course, Fall 2025, at National Taiwan University. The goal of this assignment is to develop and evaluate segmentation models for medical images using different neural network architectures and training strategies.


## 🧩 Setup
To install all required dependencies, simply run:
```
uv sync
```
> [!TIP]
> Ensure that uv is installed beforehand: `pip install uv`
If you wish to use any additional packages, please obtain approval from the TA beforehand.


## 🏋️ Training
To start training:
```
bash script/run_train.sh
```


## 📦 Generating Submission
To generate predictions for submission:
```
bash script/run_problem1.sh
```


## 🧠 Model Architecture

### Problem 1: Simple-CNN UNet

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/c9cf6b13-ce50-4e04-aafe-cec357774b30" />


### Problem 2: Trnasformer + UNet
Please include the figure of your model architecture here.
(e.g., a diagram illustrating how Transformer layers are integrated into UNet.)


## Model Configuration

### Problem 1: Simple-CNN UNet

<table>
  <tr>
    <td><b>Component</b></td>
    <td><b>Value</b></td>
  </tr>

  <tr>
    <td>Model</td>
    <td>UNetSmall (base=16)</td>
  </tr>

  <tr>
    <td>Input Size</td>
    <td>256 × 256</td>
  </tr>

  <tr>
    <td>Loss Function</td>
    <td>Dice + BCE (combined)</td>
  </tr>

  <tr>
    <td>Optimizer</td>
    <td>Adam</td>
  </tr>

  <tr>
    <td>Learning Rate</td>
    <td>0.001</td>
  </tr>

  <tr>
    <td>LR Scheduler</td>
    <td>CosineAnnealingLR (T_max = 5)</td>
  </tr>

  <tr>
    <td>Epochs</td>
    <td>5</td>
  </tr>

  <tr>
    <td>Train Batch Size</td>
    <td>8</td>
  </tr>

  <tr>
    <td>Valid Batch Size</td>
    <td>1</td>
  </tr>

  <tr>
    <td>Shuffle</td>
    <td>Train = True, Valid = False</td>
  </tr>

  <tr>
    <td>Transforms</td>
    <td>Load → ToTensor → Resize(256×256)</td>
  </tr>

  <tr>
    <td>Checkpoint Path</td>
    <td>checkpoints/cnn_unet_best.pth</td>
  </tr>
</table>


### Problem 2: Trnasformer + UNet
Provide your model configuration and hyperparameters here. (e.g., Transformer depth, attention heads, patch size, hidden dimensions, and training details, etc.)


## 🔍 Self-Supervised Learning
Describe the design of your self-supervised learning method here. You may include:
- The pretext task or objective
- Model architecture
- Loss function and training strategy
- How it benefits the downstream segmentation task


## 📈 Learning Curve
Include the following plots:
- Learning rate (per step)
- Training loss (per step)
- Validation Dice Score (per epoch)
> [!TIP]
> You can download these logs and plots directly from Weights & Biases (wandb).


## 🧪 Experiment Results

### Data Augmentation Ablation
<table>
  <tr>
    <td>Data Augmentation</td>
    <td>Validation DSC</td>
  </tr>
  <tr>
    <td>❌</td>
    <td></td>
  </tr>
  <tr>
    <td>✅</td>
    <td></td>
  </tr>
<table>

### Loss Ablations
<table>
  <tr>
    <td>Loss</td>
    <td>Validation DSC</td>
  </tr>
  <tr>
    <td>Binary Cross-Entropy</td>
    <td></td>
  </tr>
  <tr>
    <td>Dice Loss</td>
    <td></td>
  </tr>
<table>

### Model Ablations
<table>
  <tr>
    <td>Model</td>
    <td>Validation DSC</td>
    <td>Public DSC</td>
  </tr>
  <tr>
    <td>UNet + CNN</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>UNet + Transformer</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>UNet + Transformer + SSL</td>
    <td></td>
    <td></td>
  </tr>
<table>


📚 Notes
- Do not upload model checkpoints or data to GitHub repository.
- Ensure your results are reproducible by fixing random seeds and clearly specifying configurations.
- Figures, Tables, and experiment descriptions should be self-contained and interpretable without external references.
- Please maintain clarity and organization throughout your report and code structure.
