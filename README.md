# Corn Crop Disease Prediction Model - README

## Introduction

This repository contains a machine learning model developed to classify corn crop leaf images into four classes:
1. Cercospora
2. Common Rust
3. Healthy
4. Northern Leaf Blight

The model has been developed using the VGG16 architecture and further enhanced with Convolutional Block Attention Modules (CBAM) to improve its performance. The original VGG16 model, leveraged via transfer learning, was fine-tuned and compared against the CBAM-enhanced version.

## Problem Description

Corn crop diseases can lead to significant reductions in yield if not detected early. This project aims to automate the detection and classification of diseases in corn leaves using deep learning techniques, enabling farmers and agronomists to take timely action.

## Theoretical Background

### Transfer Learning
Transfer learning involves using a pre-trained model, typically trained on a large dataset like ImageNet, as a starting point for a new task. The lower layers of these models capture generic features (like edges and textures), which can be useful across different tasks. By fine-tuning the upper layers, the model can be adapted to a specific problem like disease classification in corn leaves.

### Convolutional Block Attention Module (CBAM)
CBAM is an attention mechanism that improves the representational power of a network by focusing on important features in both spatial and channel dimensions. CBAM consists of two sub-modules:
1. **Channel Attention Module:** Highlights important channels (e.g., filters).
2. **Spatial Attention Module:** Focuses on crucial spatial regions in an image.

By integrating CBAM blocks in specific layers of VGG16, the model becomes more sensitive to important regions, leading to better performance in disease detection.

## Dataset Description

The dataset contains images of corn leaves, categorized into the following four classes:
1. Cercospora
2. Common Rust
3. Healthy
4. Northern Leaf Blight

The dataset is split into training and testing sets, which are used to evaluate the model’s performance.

## Model Architectures

### Transfer Learning using VGG16
The first approach involves using VGG16 as a feature extractor. The architecture:
- Loads the pre-trained VGG16 model with `imagenet` weights, excluding the top dense layers.
- Freezes all layers of VGG16 during initial training.
- Adds a custom head with a flatten layer, dropout, dense layers, and a final output layer for classification.

### Modified VGG16 with CBAM
In this improved model:
- The CBAM blocks are inserted after specific convolutional layers (`block2_conv2`, `block3_conv3`, and `block4_conv3`) of VGG16.
- The base VGG16 model is initially frozen, the CBAM blocks and the head layers are trained.
- Finally, the entire model is fine-tuned by unfreezing the layers of VGG16 after each CBAM block.

The CBAM blocks improve the model’s attention to relevant features, enhancing overall classification accuracy.

## Performance Comparison

| Model                         | Training Accuracy | Validation Accuracy | F1 Score| Comments |
|-------------------------------|-------------------|---------------------|---------|----------|
| Transfer Learning (VGG16)     | ~99%              | ~97%                | 0.97    |Initial model using transfer learning. |
| Modified VGG16 with CBAM      | ~99%              | ~99%                | 0.99    |Enhanced model with better attention mechanism (CBAM). |

The CBAM-enhanced model outperformed the plain transfer learning model by better focusing on important features, leading to improved accuracy.

## Model Training and Evaluation

1. **Training Phases:**
   - The transfer learning model was trained for 10 epochs with VGG16 layers frozen.
   - For the CBAM-enhanced model, the CBAM blocks were initially trained with VGG16 layers frozen.
   - The entire model was then fine-tuned with a reduced learning rate for additional epochs.

2. **Evaluation:**
   - The models were evaluated using confusion matrices and classification reports (Precision, Recall and F1 score).
   - The CBAM model showed better precision, recall, and F1 scores for disease classes.

## Dataset File Structure

```
data/
    train/
        0/
        1/
        2/
        3/
    test/
        0/
        1/
        2/
        3/
```

## Conclusion

The integration of CBAM into VGG16 provides a significant boost in model performance by enhancing the model’s ability to focus on relevant spatial and channel features. This attention mechanism proves to be particularly beneficial in tasks involving subtle differences between classes, such as distinguishing between different types of leaf diseases.