# Multimodal Model Training Phases

## Introduction

This document details the training phases for a multimodal GPT model capable of processing and generating information from multiple modalities, such as text and images. It follows the technical and training insights from the project based on LLAVA 1.5 with key differences highlighted.

## Model Architecture

- **CLIP and PHI-2 Integration**: Incorporates two pretrained models, CLIP for image and text representation and PHI-2, a 2.7 billion-parameter language model.
- **Projection Layer**: A set of linear layers with residual connections, facilitating the embedding alignment between CLIP and PHI-2.

[comment]:![Multimodal GPT model architecture](path_to_figure_1_image)

## Training Strategy

Training occurs in two stages:

### Stage 1: Pretraining

1. **Objective**: Train the model to predict captions for an input image using the COCO 2017 dataset.
2. **Configuration**: Utilizes 1x A6000 GPU and CLIP-ViT-B-224px for a streamlined approach.
3. **Dataset**: COCO 2017, comprising 118,287 images and corresponding captions.

### Stage 2: Fine-tuning

1. **Objective**: Continue training the projection layer and fine-tuning PHI-2 using QLoRA for instruction following, with the model predicting the answer to a given question and an image.
2. **Configuration**: Employs 8x A6000 GPUs and CLIP-ViT-B-224px.
3. **Dataset**: Instruct150K, created via GPT-4 prompts applied to COCO images, featuring 199,770 question-answer pairs.

## Data Preparation

- **For Stage 1**: The COCO 2017 dataset is prepared to train the model in an autoregressive manner, with image embeddings remaining constant and captions undergoing sequential shifts.
  
  ![Example of image and captions from COCO 2017 dataset](path_to_figure_3_image)

- **For Stage 2**: Similar to Stage 1, but both image embeddings and questions are maintained in the input, with the shift applied to the answers.
  
  ![Examples from Instruct150K dataset](path_to_figure_5_image)

## References

- Learning Transferable Visual Models From Natural Language Supervision by Alec Radford et al., 2021.
- Improved Baselines with Visual Instruction Tuning by Liu, Haotian et al., 2023.
- QLoRA: Efficient Finetuning of Quantized LLMs by Dettmers, Tim et al., 2023.

For more detailed code, training insights, and results, refer to Part 2 of this series.
