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
bash scripts/run_train.sh
```

## Checkpoints drive link
https://drive.google.com/drive/folders/1PHAQzjuSVXZZ3hXJMS-ECtd6YRri78EI?usp=drive_link

## 📦 Generating Submission
To generate predictions for submission:
Problem 1 :
Use SmallUNet 
```
from src.model import UNetSmall
model = UNetSmall(in_ch=1, out_ch=1, base=16)
```
in generate_submission.py

```
bash scripts/run_problem1.sh --config cnn_unet
```

Problem 2 :

change in generate_submission.py (replace the model with transformer_unet)
```
from src.model_transformer_unet import build_model
model = build_model()
````
```
bash scripts/run_problem2.sh --config transformer_unet
```

Problem 3 :

Keep the same as problem 2

```
bash scripts/run_problem3.sh --config transformer_unet_ssl
```

**NOTE : Try to run the code in Colab , Already .ipynb file attached need to change some code and paths , easy for evaluation and submission file generation**

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

### Objective

Enhance the UNet baseline by integrating self-attention through a Transformer block to improve global feature modeling.

### Approach

We extend the UNet by adding a TransformerBlock inside the bottleneck:

1. Bottleneck features (B,256,28,28) are flattened into sequence form (B,784,256)

2. A 4-head Multi-Head Attention TransformerBlock processes global context

3. The output is reshaped back and decoded through UNet upsampling layers

This allows the model to capture:

- Long-range feature dependencies

- Better anatomical structure relationships

- Richer context than a purely convolutional model

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

### Objective

Improve segmentation performance by pretraining the encoder using Self-Supervised Learning (SSL) on unlabeled medical images.

### Approach (Two-Stage Training)
Stage 1 — SSL Pretraining (Denoising Autoencoder)

- Add Gaussian noise to input images

- Train an autoencoder to reconstruct the clean image

- Loss used: MSE (reconstruction loss)

- Pretrained encoder layer (enc1) is saved

Stage 2 — Supervised Segmentation (Fine-Tuning)

- Load SSL encoder weights into TransformerUNet.enc1

- Fine-tune using labeled data

- Loss used: Dice + BCE

- Leads to much better initialization than random weights

Benefits of SSL

- Learns texture, edges, and anatomical patterns before segmentation

- Improves representation quality

- Faster convergence

- Higher Dice Score

- More stable training with fewer failures

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

<img width="5056" height="2656" alt="W B Chart 11_21_2025, 11_22_18 PM" src="https://github.com/user-attachments/assets/6d093a99-6f63-4fe8-bbf8-196a8915fc6e" />


- Training loss (per step)

<img width="5056" height="2656" alt="W B Chart 11_21_2025, 11_22_29 PM" src="https://github.com/user-attachments/assets/5036d821-a611-4b88-89ce-6f9a967faec1" />

- Validation Dice Score (per epoch)

<img width="5056" height="2656" alt="W B Chart 11_21_2025, 11_20_34 PM" src="https://github.com/user-attachments/assets/95f5f877-6c4c-45d2-b395-fdd89dc4ed0a" />



## 🧪 Experiment Results

### Data Augmentation Ablation
<table>
  <tr>
    <td>Data Augmentation</td>
    <td>Validation DSC</td>
  </tr>
  <tr>
    <td>❌</td>
    <td>0.0312</td>
  </tr>
  <tr>
    <td>✅</td>
    <td>0.0320</td>
  </tr>
<table>

### Loss Ablations
<table>
  <tr>
    <td>Loss Function</td>
    <td>Train Loss</td>
    <td>Validation DSC</td>
  </tr>
  <tr>
    <td>Dice + BCE </td>
    <td>0.3068</td>
    <td>0.0320</td>
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
    <td>0.0315</td>
    <td>0.2896</td>
  </tr>
  <tr>
    <td>UNet + Transformer</td>
    <td>0.0317</td>
    <td>0.3472</td>
  </tr>
  <tr>
    <td>UNet + Transformer + SSL</td>
    <td>0.0320</td>
    <td>0.4397</td>
  </tr>
<table>


📚 Notes
- Do not upload model checkpoints or data to GitHub repository.
- Ensure your results are reproducible by fixing random seeds and clearly specifying configurations.
- Figures, Tables, and experiment descriptions should be self-contained and interpretable without external references.
- Please maintain clarity and organization throughout your report and code structure.
