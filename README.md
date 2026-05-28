# Multi-Task Traffic Sign Recognition — Robustness Under Adverse Conditions

> A systematic ablation study comparing baseline, augmented, and multi-task ResNet-18 models for traffic sign classification under seven real-world visual corruptions. Built for the Deep Learning for Computer Vision final project at USF.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)](https://pytorch.org)

---

## Why I Built This

Standard traffic sign classifiers are trained and evaluated on clean images. Real-world deployment involves rain, fog, motion blur, and poor lighting — conditions that collapse model accuracy. This project asks a concrete question: **does training with synthetic corruptions actually improve robustness, and does multi-task learning add further benefit?**

Rather than just training one model and reporting accuracy, I ran a controlled ablation across three architectures and five corruption intensities to produce a real answer.

---

## Architecture

### Three Model Variants

**1. Baseline ResNet-18**
Standard ResNet-18 pretrained on ImageNet, fine-tuned for 43-class traffic sign classification. No corruption augmentation during training.

**2. Augmented ResNet-18**
Same backbone. During training, each image has a `corruption_prob` chance of receiving one of 7 OpenCV corruptions applied at random.

**3. MultiTaskResNet18**
```
ResNet-18 Backbone (shared, frozen early layers)
        │
        ├──► Sign Head:      Linear(512 → 43)   # traffic sign class
        └──► Condition Head: Linear(512 → 8)    # corruption type
```
Total loss = `sign_loss + 0.3 × condition_loss`

The auxiliary condition head forces the backbone to learn corruption-aware features rather than memorizing clean image statistics.

### 7 Corruption Types (OpenCV)

| Corruption | Simulates |
|---|---|
| `low_brightness` | Night / underexposed camera |
| `overexposure` | Direct sunlight glare |
| `shadow` | Partial occlusion / tree shadow |
| `fog` | Low-visibility weather |
| `rain` | Wet lens / rainfall |
| `motion_blur` | Camera shake / fast movement |
| `low_contrast` | Foggy or hazy conditions |

---

## Ablation Study

Five corruption probability levels tested: `{0.0, 0.3, 0.5, 0.7, 1.0}`

| Model | Corruption Prob | Clean Acc | Corrupted Acc |
|---|---|---|---|
| Baseline | 0.0 | Highest | Lowest |
| Augmented | 0.5 | Moderate | Improved |
| Multi-Task | 0.5 | Competitive | Best overall |

*Full numerical results in notebook output cells.*

---

## Core CS Concepts

| Concept | Where Applied |
|---|---|
| **Transfer learning** | Pretrained ResNet-18 backbone fine-tuned for domain |
| **Multi-task learning** | Shared representation, dual loss weighting |
| **Data augmentation** | Stochastic corruption injection during training |
| **Ablation study** | Controlled variable isolation across 5 × 3 configurations |
| **Loss weighting** | Hyperparameter `0.3` balances auxiliary head influence |
| **OpenCV image processing** | Programmatic corruption synthesis |

---

## Setup

```bash
git clone https://github.com/Divyansh-Maurya-25/multitask-sign-recognition.git
cd multitask-sign-recognition
pip install torch torchvision opencv-python numpy matplotlib
```

Open `train_model.ipynb` in Jupyter or Google Colab and run cells sequentially. The notebook contains all three model definitions, the corruption pipeline, training loops, and evaluation.

---

## File Structure

```
multitask-sign-recognition/
├── training_submission/
│   └── train_model.ipynb    # Full training + ablation notebook
└── README.md
```

---

## Key Finding

Multi-task learning improved robustness over the augmented baseline, particularly at high corruption probabilities (`0.7`, `1.0`). The condition head acts as a regularizer that prevents the backbone from relying on texture shortcuts present only in clean images. The 5-point deduction in grading correctly identified that OpenCV-synthesized corruptions don't fully replicate real sensor degradation — a limitation worth noting for anyone extending this work with real adverse-weather datasets.

---

## Course Context

Final project for **CIS 4930 — Deep Learning for Computer Vision** at the University of South Florida. Grade: 95/100.
