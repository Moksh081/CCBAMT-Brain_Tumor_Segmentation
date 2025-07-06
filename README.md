# CCBAMT: Attention‑Guided CNN‑Transformer Hybrid for Brain Tumor Segmentation

A deep learning framework that fuses 2D CNNs, Convolutional Block Attention Modules (CBAM) and Transformer encoders to accurately detect and segment brain tumors on MRI scans (BraTS 2020).
• The BraTS 2020 dataset is utilized for accurate, efficient, and generalizable
 brain tumor detection.
 • Proposed hybrid CCBAMT approach for improving brain tumor detection ac
curacy and resilience.
 • Enhances generalization, captures long-range features, and improves segmen
tation.
 • Proposed CCBAMT achieves 99.1% accuracy and 0.792 Dice coefficient score.



## 🧠 Project Overview

Manual segmentation of brain tumors on MRI is time‑consuming and prone to variability. **CCBAMT** combines:

1. **CNN backbone** for local texture extraction  
2. **CBAM** to highlight important channels & regions  
3. **Multi‑Head Transformer** to capture global spatial dependencies  

This hybrid yields state‑of‑the‑art performance on the BraTS 2020 challenge.

---

## 📂 Repository Structure

.
├── BRAIN_TUMOR_IMAGE_SEGMENTATION.ipynb   # End‑to‑end notebook
├── model.py                               # Model definition & training script
├── methodology.png                        # Graphical overview of CCBAMT architecture
├── graphical_visualization.png            # Training & validation curves
├── mldlbaseline_comparison.png            # Baseline ML/DL performance comparison
├── layer_summary.pdf                      # Detailed layer‑by‑layer summary
├── README.md                              # ← You are here
└── requirements.txt                       # Python dependencies

---

## 📦 Dataset

- **BraTS 2020** (4 GB): Multi‑modal MRI scans (FLAIR, T1, T1ce, T2) + segmentation masks.  
- Download and unzip into a local `data/` folder; see preprocessing steps in the notebook.

---
Try out the complete pipeline on Kaggle:
https://www.kaggle.com/code/studywarriors/brats-image-segmentation-cbam-transformer-99

## ▶️ Methodology 

![image](https://github.com/user-attachments/assets/fb06ab28-1992-4764-b0da-924944f0325a)

## 📊 Results & Comparisons

![image](https://github.com/user-attachments/assets/50eba79e-fda3-47f5-b449-ea873ce6ab39)


## 📈 Visualizations

![image](https://github.com/user-attachments/assets/c1d6cf6b-0e54-47f9-af34-b37932b4e55a)
