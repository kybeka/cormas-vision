# YOLO Object Detection for Game Piece Recognition

This project implements a comprehensive YOLO-based object detection system specifically designed for recognizing game pieces in top-down board game scenarios. The system combines traditional supervised learning with advanced pseudo-labeling techniques to minimize manual annotation requirements while aiming for high detection accuracy.

## Project Overview

Game piece detection presents unique challenges: pieces can appear at various orientations, lighting conditions change throughout gameplay, and manual annotation of thousands of frames is prohibitively expensive. This project addresses these challenges through a multi-faceted approach that leverages both human expertise and machine learning automation.

The system uses oriented bounding boxes (OBB) to handle rotated boards effectively, implements iterative pseudo-labeling to reduce annotation workload, and provides interactive validation tools to maintain quality control throughout the training process.

## Key Components

**YOLO Training Pipeline**: Custom training scripts optimized for game piece detection using YOLOv11 with oriented bounding box support. The pipeline handles data augmentation, model training, and performance evaluation.

**Pseudo-Labeling System**: An advanced iterative approach that starts with minimal labeled data and progressively expands the training set using high-confidence model predictions validated through human oversight.

**Data Management**: Comprehensive tools for managing labeled and unlabeled image pools, tracking training iterations, and maintaining data quality throughout the annotation process.

**Interactive Validation**: Web-based tools for quickly reviewing and approving pseudo-labels, ensuring that automated predictions meet quality standards before being added to the training set.

## Getting Started

**Prerequisites**: Python 3.8+, CUDA-compatible GPU (recommended), initial set of labeled game images

**Installation**:
```bash
# Install required dependencies
pip install -r requirements.txt

# Prepare your initial labeled dataset
python create_unlabeled_pool.py

# Start training
python train_v2.py
```

**For Pseudo-Labeling**: Navigate to the `pseudo_labeling/` directory for the iterative pseudo-labeling pipeline that can significantly reduce your annotation workload.

## Project Structure

- `train_v2.py` - Main training script for YOLO model
- `create_unlabeled_pool.py` - Utility for preparing unlabeled image datasets
- `extract_board_coord.py` - Board coordinate extraction utilities
- `pseudo_labeling/` - Complete pseudo-labeling pipeline and tools
- `frames.v3i.yolov11/` - Training dataset in YOLO format
- `runs/` - Training outputs and model checkpoints
- `unlabeled_pool/` - Images available for pseudo-labeling

## Results

The system achieves high detection accuracy while dramatically reducing annotation time. Through iterative pseudo-labeling, we've demonstrated 50% reduction in manual labeling effort while maintaining model performance comparable to fully supervised approaches.

**Model Performance**: High precision and recall on game piece detection tasks with robust handling of various orientations and lighting conditions.

**Efficiency Gains**: Significant reduction in annotation time through intelligent pseudo-labeling and interactive validation workflows.

This approach represents a practical solution for scenarios where high-quality object detection is needed but annotation resources are limited.