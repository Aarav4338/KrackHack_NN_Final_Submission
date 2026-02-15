# KRACKHACK 2026 — Autonomous Off-Road Semantic Segmentation for Rover Navigation

Built with ❤️ for Krackhack 2026

---

# Introduction

Autonomous rover navigation in off-road environments is one of the most challenging problems in robotics and AI. Unlike structured urban roads, off-road terrain contains unpredictable hazards such as rocks, water bodies, shadows, and uneven surfaces.

To solve this, we developed a **real-time semantic segmentation system using Transformer-based SegFormer architecture** that enables rovers to understand terrain at pixel-level precision and safely navigate complex environments.

This project delivers a complete end-to-end pipeline including:

- Dataset preprocessing  
- Transformer-based model training  
- Advanced loss engineering  
- Safety auditing tools  
- Real-time demo video generation  
- Professional deployment package  

---

# Problem Statement

Autonomous off-road rovers must safely navigate terrains containing:

- Rocks (danger obstacles)
- Water bodies
- Sand and ground
- Safe objects
- Sky regions

Traditional navigation systems relying on LiDAR or basic CNN models struggle due to:

- Poor generalization
- Lighting variations
- Shadows misclassified as rocks
- Small obstacle detection failures
- Class imbalance (few rocks, lots of ground)

This creates a major safety risk.

The goal of this project was to build a **robust real-time semantic segmentation system capable of accurately detecting terrain hazards and safe regions.**

---

# Our Solution

We implemented a **Transformer-based SegFormer semantic segmentation system** optimized specifically for rover navigation.

Key capabilities:

• Pixel-level terrain understanding  
• Accurate obstacle boundary detection  
• Real-time inference (15+ FPS)  
• Robust performance under harsh lighting  
• Safety auditing visualization tools  
• Professional deployment ready system  

Final Performance:

**Mean Intersection over Union (mIoU): 64.5%**

---

# System Requirements

Minimum:

Python 3.8+  
8GB RAM  
GPU optional  

Recommended:

Python 3.10  
CUDA GPU  
16GB RAM  

---

## Complete End-to-End Workflow

---

## Phase 1: Environment & Data Setup

### Environment Initialization

We created an isolated Python virtual environment to ensure dependency stability.

Libraries used:

- torch
- albumentations
- opencv-python
- numpy
- matplotlib

---

# The Main Work

## End-to-End Workflow

## Phase 1: Environment & Data Setup

• Created isolated Python virtual environment  
• Organized dataset into images and masks  
• Built custom `FalconDataset` class  
• Applied Albumentations augmentations:

- Grid distortion  
- Elastic transform  
- Noise and brightness variation  

Purpose: Improve robustness and prevent overfitting.

---

## Phase 2: Model Architecture Design

Architecture: **SegFormer (MiT-B1)**

Why SegFormer:

- Uses Self-Attention
- Understands global context
- Better rock vs shadow differentiation
- Real-time performance (15+ FPS)

Attached lightweight MLP decoder for efficient prediction.

---

## Phase 3: Training & Optimization

Problem: Severe class imbalance.

Solution: Hybrid Loss

Loss = Dice Loss + Focal Loss

Dice Loss → improves shape accuracy  
Focal Loss → forces learning of hard examples  

Training:

- 15 epochs
- Cosine Annealing LR Scheduler
- Best model saved as:

segformer_v2_epoch_15.pth

---

## Phase 4: Validation & Safety Auditing

Evaluation using:

evaluate_model.py


Metrics:

- Mean IoU: 64.5%
- Precision
- Recall
- Confusion Matrix

Safety auditing using:

audit_failures.py


Generates Neon edge visualization to verify obstacle boundaries.

Demo generated using:

demo_video.py


Output:

results/Offroad_Segmentation_Demo.mp4

---

## Phase 5: Packaging & Submission

• Selected best model checkpoint  
• Archived intermediate models  
• Organized clean folder structure  

---

### Basic VS Code Folder Structure needed
KRACKHACK/
- .venv
- archive/
- Audit_Failures/
- Final_Results/
- Heatmap_Audit/
- Offroad_Segmentation_testImages/
- Offroad_Segmentation_Training_Dataset/
- results/

- train.py
- evaluate_model.py
- demo_video.py
- audit_failures.py

- requirements.txt
- segformer_v2_epoch_15.pth
- README.md
- .gitignore

### Youtube video link
https://youtu.be/qYacSBVCznk

### path file links 
**The 15th epoch_path_file in the drive is the main one**
https://drive.google.com/file/d/1gpC4f3CM4C-pBBe0oTsH_8eGsGSV6YkW/view?usp=sharing
