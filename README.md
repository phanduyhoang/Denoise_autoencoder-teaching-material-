# Image Denoising with Autoencoders

This repository is part of my teaching materials aimed at helping students and professors understand the practical implementation of **Autoencoders for Image Denoising**.

**Disclaimer**: This code is provided for educational purposes only. If you wish to use this project for **commercial purposes**, please reach out to me. Let's collaborate and maybe make some bucks together!

---

## Overview

Autoencoders are a type of neural network used to learn efficient representations of input data, typically for tasks like dimensionality reduction, feature extraction, or denoising. In this project, we train an **autoencoder** to remove noise from images using the **CIFAR-10 dataset**.

### Why Use Autoencoders for Denoising?
- **Learnable Noise Removal**: Autoencoders learn a mapping from noisy inputs to clean outputs, making them adaptable to specific types of noise.
- **Preservation of Image Details**: Unlike traditional filters (e.g., Gaussian blur), autoencoders can preserve fine details while removing noise.
- **Flexibility**: The same model can be trained for different types of noise, such as Gaussian noise or Salt-and-Pepper noise.
- **Data-Driven**: Autoencoders improve as they see more data, unlike hand-engineered methods that rely on fixed parameters.

---

## Libraries Used

This project uses the following libraries:
1. **TensorFlow/Keras**: For building and training the autoencoder model.
2. **NumPy**: For data manipulation and adding noise.
3. **Matplotlib**: For visualizing the denoising results.
4. **scikit-image**: For evaluating the model using metrics like PSNR and SSIM.

---

## Key Features of the Code

### 1. Dataset Preparation
- The **CIFAR-10 dataset** is loaded using TensorFlow/Keras.
- The dataset is split into **training**, **validation**, and **test sets** (80/10/10).
- Random **Gaussian noise** and **Salt-and-Pepper noise** are added to simulate noisy inputs.

### 2. Autoencoder Architecture
The autoencoder consists of:
- **Encoder**:
  - `Conv2D` layers with **Batch Normalization** and **MaxPooling** to extract features and reduce spatial dimensions.
  - A bottleneck layer with a fully connected `Dense` layer for compressed representation.
- **Decoder**:
  - `Dense` and `Reshape` layers to expand back from the bottleneck.
  - `Conv2DTranspose` layers with **Batch Normalization** and **UpSampling** to reconstruct the image.

### 3. Residual Connections
- **Skip connections** are added to the decoder to reuse spatial information from the encoder. This improves reconstruction quality and helps retain finer details.
### Residual Connections Matter!
This project demonstrates the importance of **residual connections** in autoencoders. Residuals allow the decoder to access features directly from the encoder, improving reconstruction quality. 

Try removing the residual connections in the code, and you’ll see how the results can get **real messed up**—with blurry and oversimplified outputs. Residuals make a **huge difference**!

### 4. Loss Function
- The model uses **Mean Squared Error (MSE)** as the loss function. Optionally, a **perceptual loss** combining MSE and SSIM can be used for better perceptual quality.

### 5. Training
- The model is trained with the **Adam optimizer** for 100 epochs.
- Validation loss is monitored during training to ensure the model is learning properly.

### 6. Testing and Visualization
- After training, the model is tested on the noisy test dataset.
- Results are visualized as side-by-side comparisons of:
  - Noisy Images
  - Denoised Images (output from the model)
  - Original Clean Images (ground truth)

---

## How to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/image-denoising-autoencoder.git
   cd image-denoising-autoencoder
