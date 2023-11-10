# Alzheimer Detection Classification using ResNet50

<img width="793" alt="Screenshot 2023-11-10 at 10 31 38 AM" src="https://github.com/Daanish-Hindustani/ResNet50-Alzheimer-Detection/assets/134811343/9e143a18-d24e-4fde-9615-a637024f396f">

## Overview

This repository contains code for training and evaluating a convolutional neural network (CNN) based on the ResNet50 architecture for Alzheimer detection. The model is trained on a dataset comprising 6400 images, categorized into four classes: 'Mild_Demented', 'Moderate_Demented', 'Non_Demented', and 'Very_Mild_Demented'.

## Motivation Behind the Project

The motivation for the Alzheimer Detection Classification project is driven by a curiosity-driven exploration of learning a new CNN architecture, specifically ResNet50. The goal is to apply advanced deep learning techniques to contribute meaningfully to medical research and improve the early detection of Alzheimer's Disease. The project aligns with the broader trend of utilizing AI in healthcare, providing an opportunity for personal growth and skill development in the realms of medical image analysis and deep learning. Ultimately, the project aims to leverage technology for positive impacts on patient outcomes and advance our understanding of neurodegenerative conditions.

## Model Architecture

The neural network leverages the ResNet50 architecture, a deep residual network known for its ability to train very deep neural networks successfully. ResNet50 consists of 50 layers and introduces the concept of residual learning, which helps mitigate the vanishing gradient problem.

### ResNet50 Architecture Overview

ResNet50 introduces residual blocks that contain shortcut connections, allowing the model to skip layers during training. This enables the network to learn identity mappings, making it easier to optimize and train deep networks. The architecture consists of the following key components:

1. **Convolutional Layers:** ResNet50 starts with a series of convolutional layers that extract features from the input image.

2. **Residual Blocks:** The core of ResNet50 is its residual blocks. Each block contains multiple convolutional layers and a shortcut connection that skips one or more layers. This helps in the smooth flow of gradients during backpropagation.

3. **Global Average Pooling (GAP):** Instead of using traditional fully connected layers at the end of the network, ResNet50 uses GAP to reduce spatial dimensions and parameters. This aids in better generalization and reduces overfitting.

4. **Dense Layers:** Following the ResNet50 blocks, additional dense layers are added for the final classification.

### Model Summary:

```plaintext
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 resnet50 (Functional)       (None, 2048)              23,587,712  
                                                                 
 module_wrapper_3 (ModuleWra  (None, 2048)              0         
 pper)                                                           
                                                                 
 module_wrapper_4 (ModuleWra  (None, 512)               1,049,088   
 pper)                                                           
                                                                 
 module_wrapper_5 (ModuleWra  (None, 4)                 2,052      
 pper)                                                           
                                                                 
=================================================================
Total params: 24,638,852
Trainable params: 1,051,140
Non-trainable params: 23,587,712
_________________________________________________________________
```

## Training Results

- Training Accuracy: 98%
- Training Loss: 0.06
- Testing Accuracy: 83%
- Testing Loss: 0.5

## Training Metrics

Here are the training metrics visualized using graphs:

### Training vs Validation 

<img width="557" alt="Screenshot 2023-11-10 at 10 29 26 AM" src="https://github.com/Daanish-Hindustani/ResNet50-Alzheimer-Detection/assets/134811343/c3f1ac0d-54b1-471e-9161-9a411b20c3af">

## Dataset

Overview
The Alzheimer MRI Preprocessed Dataset (128 x 128) is a collection of Magnetic Resonance Imaging (MRI) images sourced from various websites, hospitals, public repositories, and datasets. The images have been preprocessed and resized to a uniform 128 x 128 pixels. The dataset is intended for the classification of Alzheimer's Disease, providing a valuable resource for the development of accurate frameworks or architectures.

Dataset Details
Total Images: 6400 MRI images
Image Dimensions: 128 x 128 pixels
Classes:
Class - Mild Demented (896 images)
Class - Moderate Demented (64 images)
Class - Non Demented (3200 images)
Class - Very Mild Demented (2240 images)
Motive
The primary motive behind sharing this dataset is to encourage the design and development of accurate frameworks or architectures for the classification of Alzheimer's Disease. The availability of preprocessed MRI images in a standardized size facilitates the creation of robust models for diagnosis and research purposes.

References
ADNI (Alzheimer's Disease Neuroimaging Initiative)
Alzheimers.net
Kaggle - MRI and Alzheimer's Dataset
IEEE Xplore - Document: 9521165
Catalog.data.gov - Alzheimer's Disease and Healthy Aging Data
Nature - Article: s41598-020-79243-9
CORDIS - The final EPAD dataset is now available on the Alzheimer's Disease Workbench
