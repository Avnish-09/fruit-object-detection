# Fruit Object Detection using MobileNetV2

A deep learning-based fruit object detection system that identifies fruits and predicts their bounding boxes using TensorFlow, Keras, and MobileNetV2 transfer learning.

## Overview

This project implements a multi-output deep learning model for detecting fruits in images.

The model performs two tasks simultaneously:

- **Fruit Classification** — identifies the fruit as an apple, banana, or orange.
- **Bounding Box Regression** — predicts the location of the fruit within the image.

The system uses **MobileNetV2 pretrained on ImageNet** as the feature extraction backbone, followed by separate classification and bounding-box prediction heads.

## Dataset

The project uses the **Fruit Images for Object Detection** dataset from Kaggle.

- **300 annotated images**
- **3 fruit classes**
  - Apple
  - Banana
  - Orange
- Pascal VOC XML annotations
- Bounding boxes provided for each image
- Dataset license: **CC0-1.0**

The dataset is downloaded automatically using the Kaggle API.

## Methodology

The project follows these steps:

1. Download the dataset using the Kaggle API.
2. Parse Pascal VOC XML annotations.
3. Extract fruit class labels and bounding boxes.
4. Normalize bounding-box coordinates.
5. Resize images to `224 × 224`.
6. Split the dataset into training and validation sets.
7. Use MobileNetV2 pretrained on ImageNet for feature extraction.
8. Add classification and bounding-box regression heads.
9. Train the model for 15 epochs.
10. Visualize training performance.
11. Test predictions using validation images.
12. Save the trained model.

## Model Architecture

```text
Input Image
   │
   ▼
224 × 224 × 3
   │
   ▼
MobileNetV2
(ImageNet Weights)
   │
   ▼
Global Average Pooling
   │
   ▼
Dense Layer (128, ReLU)
   │
   ├───────────────────────┐
   ▼                       ▼
Classification Head      Bounding Box Head
Dense (3, Softmax)       Dense (4, Sigmoid)
   │                       │
   ▼                       ▼
Fruit Class              [xmin, ymin,
                          xmax, ymax]
