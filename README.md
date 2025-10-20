# ELE AI Algorithm Competition — Track 2
<img width="2880" height="660" alt="image" src="https://github.com/user-attachments/assets/61ec3dcb-7409-4091-8aac-bd9583454dc4" />

This repository contains the code, configuration, and notes for the Track 2 solution submitted to the ELE AI Algorithm Competition. The approach fine-tunes multimodal Qwen models using supervised fine-tuning (SFT) with LoRA and a progressive, risk-aware pipeline tailored for safety / hazard detection in images.

---

[ English | **[中文](README_zh.md)** ]
## Table of Contents
- Overview
- Hardware & Software
- Environment setup
- Data
- Data preprocessing & augmentation
- Pretrained model
- Method overview
- Key innovations
- Training workflow
- Inference (B leaderboard) workflow
- Reproducibility & scripts
- Contact

---

## Overview
We adapt Qwen multimodal models with parameter-efficient fine-tuning to classify risk levels in images and to apply targeted correction for high-risk cases. The pipeline uses a two-stage fine-tuning strategy: a broad risk classification model followed by a high-risk specialist model. Data augmentation and vision-language alignment techniques are applied to improve robustness.

---

## Hardware & Software

- Hardware (used for training)
  - GPU: vGPU 32GB
  - CPU: 25 vCPUs (Intel Xeon Platinum 8481C)
  - Memory: 90 GB DDR4
  - Storage: 30 GB system disk, 100 GB data disk

- Software stack
  - OS: Ubuntu 22.04 LTS
  - Python: 3.12
  - PyTorch: 2.3.0
  - CUDA: 12.1, cuDNN 8.9
  - NVIDIA GRID / vGPU driver

---

## Environment setup
Please see init.sh for the environment setup steps and package installation. The scripts pin required versions and prepare the dataset structure.

---

## Data
- Only the official dataset provided by the ELE competition organizers (饿了么) was used for training and evaluation.
- No external datasets or extra-labeling sources were incorporated.

---

## Data preprocessing & augmentation
- Preprocessing scripts and examples are provided in data-pre.sh.
- We use multiple augmentation strategies to enrich limited image data:
  - Standard geometric transformations (scaling, random rotation, cropping)
  - A multimodal adversarial augmentation suite (described below) to increase robustness while preserving semantic consistency
- Augmented image–label pairs are filtered using a semantic-consistency scoring step to reduce noisy augmentations.

---

## Pretrained model
- Base model: Qwen / Qwen2.5-VL-7B-Instruct
  - Available at: https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct
- We use LoRA for parameter-efficient supervised fine-tuning (SFT) to adapt the pretrained weights.

---

## Method overview
- Stage 1 — General risk classification:
  - Fine-tune Qwen2.5-VL-7B with LoRA (qwen2.5-vl-7b-sft) to predict coarse-grained labels:
    - no risk, low risk, medium risk, high risk, non-corridor (or other domain labels)
- Stage 2 — High-risk specialization:
  - Fine-tune a smaller, targeted model (qwen2.5-vl-3b-sft-high-risk) on high-risk examples with specialized augmentation and attention reweighting to improve recall and calibration for hazardous cases.
- Prediction combination:
  - Use a confidence-correction fusion: final_prediction = base_model_confidence × high_risk_correction_coefficient

---

## Key innovations

1. Progressive Risk-Specific Fine-Tuning
   - Two-stage framework: baseline coarse classification → targeted high-risk correction.
   - This allows the system to preserve generalization while putting extra modeling capacity on dangerous cases.
   - Introduces a High-Risk Feature Amplifier module in the specialist model to boost sensitivity to visual cues associated with hazards (spatial layouts, object co-occurrences, signage).

2. Vision-Language Adversarial Data Augmentation Suite (VL-ADAS)
   - Multimodal augmentation combining image transforms with semantic constraints:
     - Semantic-Consistent Color Jitter: operate in HSV space while preserving key hazard-colors (e.g., warning signs).
     - Context-Aware Padding: padding strategies selected based on semantic content (edge replication, reflection, or synthetic hazard-region insertion).
     - Cross-modal consistency verification: filter augmented samples by checking their image-text semantic alignment score to avoid label drift.

3. Vision-Language Alignment Correction
   - Reweight attention for high-risk features to improve the model’s focus on hazard indicators.
   - Combine confidences from the base and high-risk models to compute a calibrated final prediction.

---

## Training workflow
- Scripts: see train.sh for the full training pipeline.
- Steps:
  1. Prepare environment (init.sh) and preprocess data (data-pre.sh).
  2. Train qwen2.5-vl-7b-sft for coarse multi-class classification.
  3. Extract or sample high-risk examples and apply targeted augmentations.
  4. Train qwen2.5-vl-3b-sft-high-risk with specialized loss weighting and the High-Risk Feature Amplifier.
  5. Evaluate and iterate: perform model correction and calibration between stages.

---

## Inference (B leaderboard) workflow
- Scripts: see test.sh.
- For the B leaderboard, we run inference with both models:
  - Generate predictions from qwen2.5-vl-7b-sft (base).
  - Generate high-risk predictions from qwen2.5-vl-3b-sft-high-risk.
  - Apply the confidence fusion / correction strategy to produce the final submission file b-submit.txt.
- Reported leaderboard result: approximately 5.6938 (as obtained in our evaluation runs).

---

## Reproducibility & scripts
- init.sh — environment setup and dependency installation
- data-pre.sh — data preprocessing, augmentation, and filtering
- train.sh — training pipeline (stage 1 and stage 2)
- test.sh — inference pipeline and submission generation

To reproduce results:
1. Run init.sh to prepare environment.
2. Run data-pre.sh to prepare and augment the dataset.
3. Run train.sh to train both models (adjust paths/configs as needed).
4. Run test.sh to produce b-submit.txt and evaluate.

---

## Notes & limitations
- The approach focuses exclusively on the official competition dataset; no external data was used.
- High-risk specialization improves sensitivity but may require careful calibration to avoid increasing false positives.
- Hardware & runtime may vary; LoRA reduces GPU memory footprint but large models still require substantial resources.

---

## Contact
For questions about reproducibility, experiments, or scripts, open an issue in this repository or contact the author (repository owner).
