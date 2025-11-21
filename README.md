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

### Objective

Build a baseline medical image segmentation model using a lightweight CNN-based UNet.

### Approach

- We designed a 3-level UNet with:

- Convolutional encoder

- Symmetric decoder

- Skip connections

- 1×1 convolution output head

The model is computationally lightweight (base=16 channels), making it suitable for CPU execution while still capturing essential multi-scale features.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/c9cf6b13-ce50-4e04-aafe-cec357774b30" />


### Problem 2: Trnasformer + UNet

<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/4835be3a-5817-4f3b-804b-5ee5851a416a" />



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

<table>
  <tr>
    <td><b>Component</b></td>
    <td><b>Value</b></td>
  </tr>

  <tr>
    <td>Model</td>
    <td>TransformerUNet (base=32, 1 Transformer Block)</td>
  </tr>

  <tr>
    <td>Input Size</td>
    <td>224 × 224</td>
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
    <td>0.0001</td>
  </tr>

  <tr>
    <td>LR Scheduler</td>
    <td>CosineAnnealingLR (T_max = 50)</td>
  </tr>

  <tr>
    <td>Epochs</td>
    <td>50</td>
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
    <td>Transforms</td>
    <td>Load → ToTensor → Resize(224×224)</td>
  </tr>

  <tr>
    <td>Special Step</td>
    <td>Flatten features → TransformerBlock → reshape back</td>
  </tr>

  <tr>
    <td>Checkpoint Path</td>
    <td>checkpoints/transformer_unet_best.pth</td>
  </tr>
</table>



## 🔍 Self-Supervised Learning


<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/15378d10-ffd9-4e0f-bc2e-22f460142e80" />



<table>
  <tr>
    <td><b>Component</b></td>
    <td><b>Description</b></td>
  </tr>

  <tr>
    <td>Pretext Task</td>
    <td>Denoising Autoencoder (reconstruct clean image from noisy)</td>
  </tr>

  <tr>
    <td>Architecture</td>
    <td>3-level Conv Autoencoder, Enc1 matches TransformerUNet Enc1</td>
  </tr>

  <tr>
    <td>Loss Function</td>
    <td>MSELoss</td>
  </tr>

  <tr>
    <td>Training</td>
    <td>20 epochs, Adam(1e-3), Gaussian noise σ = 0.1, dataset without labels</td>
  </tr>

  <tr>
    <td>Knowledge Transfer</td>
    <td>Copy SSL enc1 → TransformerUNet.enc1 for fine-tuning</td>
  </tr>

  <tr>
    <td>Benefit</td>
    <td>Better initialization, faster convergence, higher DSC</td>
  </tr>
</table>



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
